
def fetch_model_class(model_type):
    if model_type == 'vltm':
        from VLTM import VLTM
    return VLTM

