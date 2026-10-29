"""Gradio demo for the SDXL Mithila / Madhubani Style LoRA on Hugging Face Spaces (ZeroGPU)
with automatic fallback to ``enable_model_cpu_offload()`` on standard GPUs or CPU environments.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr

try:
    import spaces  # type: ignore[import-not-found]

    _HAS_SPACES = True
except ImportError:
    spaces = None
    _HAS_SPACES = False

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_MODEL = os.environ.get("VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix")
LORA_REPO = os.environ.get("LORA_REPO", "raju-ai/sdxl-mithila-lora")
TRIGGER_PREFIX = "in sks_mithila style"

EXAMPLE_PROMPTS = [
    [
        "in sks_mithila style, two peacocks perched on a flowering kadamba tree branch surrounded by lotus blossoms and an ornate geometric border",
        0.85,
        28,
        7.5,
        42,
    ],
    [
        "in sks_mithila style, an astronaut riding a white horse across the cratered surface of the moon under a starry cosmos",
        0.85,
        28,
        7.5,
        12,
    ],
    [
        "in sks_mithila style, a royal procession with a decorated elephant, musicians, and dancers in intricate kachni double-line hatching",
        0.85,
        28,
        7.5,
        7,
    ],
    [
        "in sks_mithila style, a serene pond filled with golden fish, turtles, and blooming pink lotus flowers in bold bharni flat pigment fills",
        0.90,
        28,
        7.5,
        1,
    ],
    [
        "in sks_mithila style, a cozy cafe table with a tabby cat sleeping beside a steaming teacup and an open book by a window",
        0.85,
        28,
        7.5,
        24,
    ],
]


def _gpu_decorator(duration: int = 30) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if _HAS_SPACES and spaces is not None:
        return spaces.GPU(duration=duration)
    return lambda fn: fn


def _init_pipeline() -> StableDiffusionXLPipeline | None:
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        return None
    is_zerogpu = _HAS_SPACES and bool(os.environ.get("SPACE_ID") or os.environ.get("SPACES_ZERO_GPU"))
    has_cuda = torch.cuda.is_available() or is_zerogpu
    dtype = torch.float16 if has_cuda else torch.float32

    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype)
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        vae=vae,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        use_karras_sigmas=True,
    )
    try:
        pipeline.load_lora_weights(LORA_REPO)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Warning: could not load LoRA weights from {LORA_REPO}: {exc}")

    if is_zerogpu:
        # Eager module-scope .to("cuda") required for ZeroGPU tensor hijack
        pipeline.to("cuda")
    elif has_cuda:
        pipeline.enable_model_cpu_offload()
    return pipeline


pipe = _init_pipeline()


def ensure_trigger(prompt: str, auto_prepend: bool = True) -> str:
    text = prompt.strip()
    if not auto_prepend or not text:
        return text
    if TRIGGER_PREFIX.lower() not in text.lower() and "sks_mithila" not in text.lower():
        return f"{TRIGGER_PREFIX}, {text}"
    return text


@_gpu_decorator(duration=30)
def generate_image(
    prompt: str,
    lora_scale: float = 0.85,
    steps: int = 28,
    guidance_scale: float = 7.5,
    seed: int = 42,
    auto_trigger: bool = True,
):
    global pipe
    if pipe is None:
        os.environ.pop("SKIP_MODEL_LOAD", None)
        pipe = _init_pipeline()
    assert pipe is not None

    full_prompt = ensure_trigger(prompt, auto_prepend=auto_trigger)
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(int(seed))
    img = pipe(
        prompt=full_prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=gen,
        cross_attention_kwargs={"scale": float(lora_scale)},
    ).images[0]
    return img


with gr.Blocks(title="SDXL Mithila / Madhubani Style LoRA") as demo:
    gr.Markdown(
        "# 🎨 SDXL Mithila (Madhubani) Folk Art Generator\n\n"
        "Generate authentic **Mithila / Madhubani folk art** (`in sks_mithila style`) with SDXL 1.0. "
        "Captures traditional double-line *kachni* linework, bold *bharni* natural pigment fills, and floral/geometric borders.\n\n"
        "Model Weights & Evaluation: [🤗 `raju-ai/sdxl-mithila-lora`](https://huggingface.co/raju-ai/sdxl-mithila-lora)"
    )
    with gr.Row():
        with gr.Column(scale=3):
            prompt_in = gr.Textbox(
                label="Prompt",
                placeholder="in sks_mithila style, two peacocks perched on a flowering kadamba tree...",
                value=EXAMPLE_PROMPTS[0][0],
                lines=3,
            )
            with gr.Row():
                lora_scale_in = gr.Slider(0.0, 1.2, value=0.85, step=0.05, label="LoRA Scale (0.0 = Base SDXL)")
                steps_in = gr.Slider(15, 40, value=28, step=1, label="Inference Steps")
            with gr.Row():
                cfg_in = gr.Slider(3.0, 12.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                seed_in = gr.Number(value=42, precision=0, label="Seed")
                auto_trig_in = gr.Checkbox(value=True, label="Auto-prepend 'in sks_mithila style'")
            run_btn = gr.Button("Generate Artwork", variant="primary")
        with gr.Column(scale=3):
            img_out = gr.Image(label="Generated Artwork (1024×1024)", type="pil")

    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=[prompt_in, lora_scale_in, steps_in, cfg_in, seed_in],
    )
    run_btn.click(
        fn=generate_image,
        inputs=[prompt_in, lora_scale_in, steps_in, cfg_in, seed_in, auto_trig_in],
        outputs=img_out,
        api_name="predict",
    )

if __name__ == "__main__":
    demo.launch()
