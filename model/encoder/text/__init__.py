


def fetch_text_encoders(model_name):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        from .clip import ClipTextEncoder
        return ClipTextEncoder(), 512
    return None, None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        from .clip import ClipTokenizer
        return ClipTokenizer()
    elif model_name == 'PaliGemma':
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    else:
        raise NotImplementedError(f"Tokenizer {model_name} not implemented.")

def fetch_pretrained_model(name):
    if name == 'PaliGemma':
<<<<<<< HEAD
        from .paligemma_with_expert import (
=======
        from paligemma_with_expert import (
>>>>>>> 6e963d0 (v-0.0.1  pi0改动 touch encoder 和 2D 视觉编码器，修改动作预测头以适应新的动作维度)
            PaliGemmaWithExpertConfig,
            PaliGemmaWithExpertModel,
        )
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=True,
            train_expert_only=True,
            attention_implementation='eager',
        )
        return PaliGemmaWithExpertModel(paligemma_with_export_config)