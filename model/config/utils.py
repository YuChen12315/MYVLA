import torch
import logging

def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using cuda.")
        return torch.device("mps")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)
    match try_device:
        case "cuda":
            assert torch.cuda.is_available()
            device = torch.device("cuda")
        case "mps":
            assert torch.backends.mps.is_available()
            device = torch.device("mps")
        case "cpu":
            device = torch.device("cpu")
            if log:
                logging.warning("Using CPU, this will be slow.")
        case _:
            device = torch.device(try_device)
            if log:
                logging.warning(f"Using custom {try_device} device.")

    return device

def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu.")
