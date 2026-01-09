def fetch_touch_encoders(model_name):
    if model_name.lower() == "moevt":
        from .Moevt import Moevt
        return Moevt
    return None