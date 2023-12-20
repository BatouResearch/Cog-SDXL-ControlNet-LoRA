import torch
from diffusers import AutoencoderKL, DiffusionPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)


CONTROL_CACHE = "control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
DPO_UNET_CACHE = "./dpo-cache"

better_vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=better_vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(SDXL_MODEL_CACHE, safe_serialization=True)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.save_pretrained(REFINER_MODEL_CACHE, safe_serialization=True)

safety = StableDiffusionSafetyChecker.from_pretrained(
    "CompVis/stable-diffusion-safety-checker",
    torch_dtype=torch.float16,
)
safety.save_pretrained(SAFETY_CACHE)

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
controlnet.save_pretrained(CONTROL_CACHE)

unet_id = "mhdang/dpo-sdxl-text2image-v1"
unet = UNet2DConditionModel.from_pretrained(
    unet_id,
    subfolder="unet",
    torch_dtype=torch.float16,
    cache_dir="unet-cache"
)
unet.save_pretrained(DPO_UNET_CACHE, safe_serialization=True)