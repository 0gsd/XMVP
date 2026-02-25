from PIL import Image
from diffusers import FluxImg2ImgPipeline
import torch

try:
    pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    pipe(prompt="test", image=None, height=256, width=256)
except Exception as e:
    print("ERROR:", type(e), e)
