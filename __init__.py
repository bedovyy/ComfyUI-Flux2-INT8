"""
int88 - Fast INT8 Tensorwise Quantization for ComfyUI

Provides:
- Int8TensorwiseOps: Custom operations for direct int8 weight loading
- OTUNetLoaderW8A8: Load int8 quantized diffusion models
- OTCheckpointLoaderW8A8: Load int8 quantized checkpoints

Uses torch._int_mm for blazing fast inference.
"""

# Register the int8_tensorwise format with ComfyUI's quant registry
# This is for metadata compatibility when saving/loading models
try:
    from comfy.quant_ops import QUANT_ALGOS, register_layout_class, QuantizedLayout
    import torch

    class Int8TensorwiseLayout(QuantizedLayout):
        """Minimal layout class to satisfy ComfyUI's registry requirements."""
        class Params:
            def __init__(self, scale=None, orig_dtype=None, orig_shape=None, **kwargs):
                self.scale = scale
                self.orig_dtype = orig_dtype
                self.orig_shape = orig_shape
            
            def clone(self):
                import torch
                return Int8TensorwiseLayout.Params(
                    scale=self.scale.clone() if isinstance(self.scale, torch.Tensor) else self.scale,
                    orig_dtype=self.orig_dtype,
                    orig_shape=self.orig_shape
                )

        @classmethod
        def state_dict_tensors(cls, qdata, params):
            return {"": qdata, "weight_scale": params.scale}
        
        @classmethod  
        def dequantize(cls, qdata, params):
            return qdata.float() * params.scale

    register_layout_class("Int8TensorwiseLayout", Int8TensorwiseLayout)

    cur_config = QUANT_ALGOS.get("int8_tensorwise")
    if cur_config is None:
        QUANT_ALGOS["int8_tensorwise"] = {
            "storage_t": torch.int8,
            "parameters": {"weight_scale", "input_scale"},
            "comfy_tensor_layout": "Int8TensorwiseLayout",
        }
    else:
        cur_config["comfy_tensor_layout"] = "Int8TensorwiseLayout"
        cur_config["parameters"] = {"weight_scale", "input_scale"}

except ImportError:
    pass

# Export the custom ops class for external use
try:
    from .int8_quant import Int8TensorwiseOps
except ImportError:
    Int8TensorwiseOps = None

from .int8_unet_loader import UNetLoaderINTW8A8

NODE_CLASS_MAPPINGS = {
    "OTUNetLoaderW8A8": UNetLoaderINTW8A8,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OTUNetLoaderW8A8": "Load Diffusion Model INT8 (W8A8)",
}
