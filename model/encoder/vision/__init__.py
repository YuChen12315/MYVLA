
def fetch_visual_encoders(model_name):
    if model_name == "clip":
        from .clip import load_clip
        return load_clip()
    elif model_name == "dino_sigl_vit":
        from .backbone2d import Backbone2D
        return Backbone2D.init("dinosiglip")
    return None
