# wuking 
# UTC current time is 08:16:07
# UTC current date is 14th Tuesday January 2025.
# UTC: Coordinated Universal Time
import gc
import os
from typing import TypeAlias

import torch
from PIL.Image import Image
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL, AutoencoderTiny
from huggingface_hub.constants import HF_HUB_CACHE
from pipelines.models import TextToImageRequest
from torch import Generator
from transformers import T5EncoderModel, CLIPTextModel
import time
from torchao.quantization import quantize_, int8_weight_only, int8_dynamic_activation_int8_weight

Pipeline: TypeAlias = FluxPipeline

CHECKPOINT = "black-forest-labs/FLUX.1-schnell"
REVISION = "741f7c3ce8b383c54771c7003378a50191e9efe9"


def load_pipeline() -> Pipeline:
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        REVISION=REVISION,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cpu")

    quantize_(pipeline.transformer, int8_weight_only())
    pipeline.transformer = pipeline.transformer.to("cuda")

    pipeline.transformer.to(memory_format=torch.channels_last)
    pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)

    pipeline.text_encoder = pipeline.text_encoder.to("cuda")
    # quantize_(pipeline.text_encoder, int8_weight_only())

    pipeline.text_encoder_2 = pipeline.text_encoder_2.to("cuda")
    # quantize_(pipeline.text_encoder_2, int8_weight_only())

    pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=torch.bfloat16)
    pipeline.vae = pipeline.vae.to("cuda")
    # quantize_(pipeline.vae, int8_weight_only())
    
    for _ in range(3):
        begin = time.time()
        pipeline(prompt="Platyhelmia, timbale, pothery, intracerebellar, actinogram, electrovital, taxis, hereness", width=1024, height=1024, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256)
        print(f"Time: {time.time() - begin:.2f}s")
    return pipeline

def infer(request: TextToImageRequest, pipeline: Pipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed)

    return pipeline(
        request.prompt,
        generator=generator,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        height=request.height,
        width=request.width,
    ).images[0]
