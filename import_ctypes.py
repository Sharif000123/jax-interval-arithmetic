import ctypes
from pathlib import Path
import jax

path = next(Path("ffi").glob("librms_norm*"))
rms_norm_lib = ctypes.cdll.LoadLibrary(path)
jax.ffi.register_ffi_target(
    "rms_norm", jax.ffi.pycapsule(rms_norm_lib.RmsNorm), platform="cpu")