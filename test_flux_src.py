import inspect
from diffusers import FluxImg2ImgPipeline

print(inspect.getsource(FluxImg2ImgPipeline.__call__))
