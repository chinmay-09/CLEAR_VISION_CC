from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Define image transform (same as training)
image_size = 256
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert uploaded PIL image to normalized tensor suitable for the model.
    """
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Shape: [1, 3, 256, 256]

def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a torch.Tensor back to a PIL image for display.
    """
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1)
    array = tensor.permute(1, 2, 0).numpy()  # [H, W, C]
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)
