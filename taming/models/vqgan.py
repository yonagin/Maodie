import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main import instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

# Maodie adversarial training components
class DirichletDiscriminator(torch.nn.Module):
    """判别器：区分真实Dirichlet样本和生成的软分配h"""
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        
    def forward(self, h):
        """h: (B, K) 概率向量"""
        return self.net(h).squeeze(-1)

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 enable_maodie=False,  # 新增参数：是否启用maodie对抗训练
                 temperature=1.0,  # maodie参数：温度参数
                 dirichlet_alpha=0.1,  # maodie参数：Dirichlet分布参数
                 lambda_adv=1e-4,  # maodie参数：对抗损失权重
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        # Maodie相关参数
        self.enable_maodie = enable_maodie
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.lambda_adv = lambda_adv
        self.num_embeddings = n_embed
        
        # Maodie组件
        if self.enable_maodie:
            self.discriminator = DirichletDiscriminator(n_embed)
        
        # 用于控制codebook_usage打印频率的计数器
        self.codebook_usage_print_counter = 0
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, h, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec
    
    def sample_dirichlet_prior(self, batch_size):
        """从Dirichlet先验中采样"""
        alpha = torch.full((self.num_embeddings,), self.dirichlet_alpha)
        dirichlet_dist = torch.distributions.Dirichlet(alpha)
        samples = dirichlet_dist.sample((batch_size,))
        return samples.to(self.device)

    def forward(self, input, return_p=False):
        quant, diff, h, info = self.encode(input)  
        dec = self.decode(quant)
        if return_p:
            # 计算与码本的距离
            z_flattened = h.view(-1, self.quantize.e_dim)
            distances = torch.cdist(z_flattened, self.quantize.embedding.weight)
            
            # 使用温度参数计算软分配
            p_soft = F.softmax(-distances / self.temperature, dim=-1)
            p_soft = p_soft.view(h.shape[0], h.shape[2], h.shape[3], -1)
            
            # 全局平均池化得到(B, K)的概率向量
            p_global = p_soft.mean(dim=(1, 2))
            return dec, diff, p_global, info
        else:
            return dec, diff , info
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        # Maodie对抗训练逻辑
        if self.enable_maodie:
            if optimizer_idx in [0, 1]: 
                xrec, qloss, p_fake, info = self(x, return_p=True)
            else:  # 原始判别器训练
                xrec, qloss, info = self(x)
            
            if optimizer_idx == 0:
                # Maodie判别器训练（第一个优化器）
                self.discriminator.requires_grad_(True)
                p_real = self.sample_dirichlet_prior(x.size(0))
                
                d_fake = self.discriminator(p_fake.detach())
                d_real = self.discriminator(p_real)
                
                # 判别器损失：最大化log(D(real)) + log(1 - D(fake))
                # 判别器输出是实数，使用BCEWithLogitsLoss
                d_loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                d_loss_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake
                # 统一记录对抗损失
                dir_losses = {
                    'train/aeloss': torch.tensor(0.0),
                    "train/dir_d_loss": d_loss,
                    "train/dir_g_loss": torch.tensor(0.0)  # 判别器训练时生成器损失为0
                }
                self.log_dict(dir_losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                return d_loss
                
            elif optimizer_idx == 1:
                self.discriminator.requires_grad_(False)  # 冻结判别器参数
                # 生成器训练（VQ-VAE + 对抗损失，第二个优化器）
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                
                # 计算对抗损失 - 使用之前计算的p_fake，避免重复计算
                d_fake = self.discriminator(p_fake)
                # 使用BCEWithLogitsLoss，生成器希望判别器输出接近1
                g_loss = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
                
                total_loss = aeloss + self.lambda_adv * g_loss
            
                # 合并显示生成器和判别器损失
                dir_losses = {
                    'train/aeloss': aeloss,
                    "train/dir_g_loss": g_loss,
                    "train/dir_d_loss": torch.tensor(0.0)
                }
                self.log_dict(['train/total_loss']) = total_loss
                self.log_dict(dir_losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return total_loss
                
            elif optimizer_idx == 2:
                # 原始判别器训练（第三个优化器，仅当存在判别器时）
                if hasattr(self.loss, 'discriminator') and self.loss.discriminator is not None:
                    discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                                        last_layer=self.get_last_layer(), split="train")
                    self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                    self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                    return discloss
                else:
                    # 如果没有原始判别器，返回0损失
                    return torch.tensor(0.0, requires_grad=True, device=self.device)
        

        else:
            result = self(x)
            if len(result) == 3:
                xrec, qloss, info = result
            else:
                xrec, qloss = result[:2]  # 只取前两个值
                info = result[2] if len(result) > 2 else None
            
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

        perplexity, _, _, codebook_usage = info
        
        # 每隔100个step打印一次codebook_usage
        self.codebook_usage_print_counter += 1
        if self.codebook_usage_print_counter % 100 == 0:
            # 将小数转换为百分比格式
            codebook_usage_percent = codebook_usage * 100
            print(f"Step {self.global_step}: Codebook Usage = {codebook_usage_percent:.2f}%")


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
    
        xrec, qloss, _= self(x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        
        if self.enable_maodie:
            # Maodie模式：按照标准GAN训练顺序，先判别器后生成器
            optimizers = []
            
            # Maodie判别器优化器（第一个）
            opt_disc_maodie = torch.optim.Adam(self.discriminator.parameters(),
                                              lr=lr, betas=(0.5, 0.9))
            optimizers.append(opt_disc_maodie)
            
            # 自编码器优化器（第二个）
            optimizers.append(opt_ae)
            
            # 检查是否有原始判别器，如果有则添加第三个优化器
            if hasattr(self.loss, 'discriminator') and self.loss.discriminator is not None:
                opt_disc_original = torch.optim.Adam(self.loss.discriminator.parameters(),
                                                    lr=lr, betas=(0.5, 0.9))
                optimizers.append(opt_disc_original)
            
            return optimizers, []
        else:
            # 原始模式：两个优化器
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
            return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_input(self, batch, k):
        """从batch中提取特定键的数据并进行预处理"""
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        # 根据Maodie模式处理forward方法的返回值
        xrec, _, _ = self(x)
        
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        
        # 根据Maodie模式处理forward方法的返回值
        xrec, qloss,info = self(x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        
        xrec, qloss, info = self(x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        
        # 根据Maodie模式处理forward方法的返回值
        xrec, _, _= self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        
        # 根据Maodie模式处理forward方法的返回值
        xrec, qloss, info = self(x)
        
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        
        xrec, qloss, info = self(x)
        
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []