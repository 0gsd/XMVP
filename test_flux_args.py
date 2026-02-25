from diffusers import FluxImg2ImgPipeline
import inspect

sig = inspect.signature(FluxImg2ImgPipeline.__call__)
print("FluxImg2ImgPipeline args:", list(sig.parameters.keys()))
