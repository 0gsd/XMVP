import sys
from diffusers import FlowMatchEulerDiscreteScheduler
import inspect

try:
    src = inspect.getsource(FlowMatchEulerDiscreteScheduler.scale_noise)
    print("SOURCE:\n", src)
except Exception as e:
    print("Error:", e)
    print("Hasattr scale_noise:", hasattr(FlowMatchEulerDiscreteScheduler, "scale_noise"))
