import torch
from diffusers import StableDiffusionXLPipeline
import gc
from PIL import Image

model_path = "model/NoobAI-XL-v1.1.safetensors"
lora_path = "model/memories-of-phantasm-style-noobai-vpred10-logit.safetensors"

_pipe = None

def get_sdxl_pipeline(offload_cpu: bool):
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
        )
        if offload_cpu:
            _pipe.enable_model_cpu_offload()

        _pipe.load_lora_weights(lora_path, adapter_name='my-lora')
        _pipe.set_adapters(["my-lora"], adapter_weights=[1])
    return _pipe


def image_generating(base_prompt: str, image_name: str, offload_cpu: bool):
    """Generates pictures for the fleshcards, based on the """
    pipe = get_sdxl_pipeline(offload_cpu)

    quality_tags = "masterpiece, best quality, perfect quality, absurdres, newest, very aesthetic"

    prompt = f"memoriesofphantasm, 1girl, cute, anime style, flat color, pastel background, simple vector art, {base_prompt}, {quality_tags}"

    negative_prompt = "photorealistic, 3d, shadows, bad anatomy, complex background, detailed, blurry, english text, Signature, 3D, pixel, watermark, fused fingers, mutation, amputee, bar censor, censored,  monochrome, greyscale, mosaic censoring, cropped, comic, blank speech bubble"

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=28,
        guidance_scale=6.0,
        width=1024,
        height=1024
    ).images[0]

    image = image.resize((512, 512), Image.Resampling.LANCZOS)

    image.save('images/' + image_name)


def unload_sdxl():
    global _pipe
    if _pipe is not None:
        _pipe.to("cpu")
        del _pipe
        _pipe = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()