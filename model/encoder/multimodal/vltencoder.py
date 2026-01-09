import torch
from torch import nn
import math
from model.config.VLTM import EncoderConfig
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders, fetch_pretrained_model, fetch_tokenizers
from ..touch import fetch_touch_encoders

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):  # 直接传入配置对象
        super().__init__()
        self.encoder = []
        self.config = config
        if config.vl_backbone:
            self.VL_encoder = fetch_pretrained_model(config.vl_backbone)
            self.encoder.append(self.VL_encoder)
        if config.touch_encoder is not None:
            self.touch_encoder = fetch_touch_encoders(config.touch_encoder)
            self.encoder.append(self.touch_encoder)
            # 设置参数是否需要梯度
            for p in self.touch_encoder.parameters():
                p.requires_grad = config.finetune_touch_encoder
        
    def forward(self, observations: dict):
        images = observations['images']
        img_masks = observations['img_masks']
        lang_tokens = observations['lang_tokens']
        lang_masks = observations['lang_masks']
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )        

        return prefix_embs, prefix_pad_masks, prefix_att_masks

        
        
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []

        # TODO: remove for loop
        for (img, img_mask) in zip(images, img_masks, strict=False):
            img_emb = self.VL_encoder.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        lang_emb = self.VL_encoder.embed_language_tokens(lang_tokens)

        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        
        
        # # TODO: touch encoder forward
        # if self.config.touch_encoder is not None:
        #     touch_embs, touch_mask, touch_att_mask = self.touch_encoder(touch_obs)
        #     embs.append(touch_embs)
        #     pad_masks.append(touch_mask)
        #     num_touch_embs = touch_embs.shape[1]
        #     att_masks += [0] * num_touch_embs
            
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks
    