"""Gradio demo for the SDXL Mithila / Madhubani Style LoRA on Hugging Face Spaces.

Supports @spaces.GPU (ZeroGPU) when deployed on Hugging Face Spaces and automatically
falls back to enable_model_cpu_offload() or CPU execution on Kaggle / Colab / local machines.
"""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

try:
    import spaces  # type: ignore[import-not-found]

    gpu_decorator = spaces.GPU(duration=60)
except ImportError:

    def gpu_decorator(fn: Any) -> Any:
        return fn


BASE_MODEL = os.environ.get("BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_MODEL = os.environ.get("VAE_MODEL", "madebyollin/sdxl-vae-fp16-fix")
LORA_REPO = os.environ.get("LORA_REPO", "raju-ai/sdxl-mithila-lora")
TRIGGER_PREFIX = "in sks_mithila style"

EXAMPLE_PROMPTS = [
    [
        "in sks_mithila style, two peacocks perched on a flowering kadamba tree branch surrounded by lotus blossoms and an ornate geometric border",
        0.85,
        25,
        7.5,
        42,
    ],
    [
        "in sks_mithila style, a snowy Himalayan village beneath Mount Everest with terraced fields, yak herders, and fluttering prayer flags",
        0.90,
        25,
        7.5,
        7,
    ],
    [
        "in sks_mithila style, an astronaut riding a white horse across the cratered surface of the moon under a starry cosmos",
        0.85,
        25,
        7.5,
        12,
    ],
    [
        "in sks_mithila style, a serene pond filled with golden fish, turtles, and blooming pink lotus flowers in bold bharni flat pigment fills",
        0.95,
        25,
        7.5,
        1,
    ],
]

_PIPE = None


def get_pipeline():
    """Lazy-load SDXL + fp16-fix VAE + Mithila LoRA weights."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import (
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        StableDiffusionXLPipeline,
    )

    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32

    vae = AutoencoderKL.from_pretrained(VAE_MODEL, torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if has_cuda else None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

    try:
        pipe.load_lora_weights(LORA_REPO)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Warning: could not load LoRA weights from {LORA_REPO}: {exc}")

    if has_cuda:
        if "SPACE_ID" in os.environ:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
    _PIPE = pipe
    return _PIPE


def ensure_trigger(prompt: str, auto_prepend: bool = True) -> str:
    text = prompt.strip()
    if not auto_prepend or not text:
        return text
    if TRIGGER_PREFIX.lower() not in text.lower() and "sks_mithila" not in text.lower():
        return f"{TRIGGER_PREFIX}, {text}"
    return text


@gpu_decorator
def generate_image(
    prompt: str,
    lora_scale: float = 0.85,
    steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    auto_trigger: bool = True,
):
    import torch

    pipe = get_pipeline()
    full_prompt = ensure_trigger(prompt, auto_prepend=auto_trigger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(int(seed))
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
        "# 🎨 SDXL Mithila (Madhubani) Style LoRA\n"
        "Generate images in traditional **Mithila / Madhubani folk art** style (`in sks_mithila style`) "
        "featuring intricate *kachni* double-line hatching, bold *bharni* pigment fills, and ornate geometric borders. "
        "Adjust **LoRA Scale** to `0.0` to compare against base SDXL 1.0."
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
                steps_in = gr.Slider(15, 40, value=25, step=1, label="Inference Steps")
            with gr.Row():
                cfg_in = gr.Slider(3.0, 12.0, value=7.5, step=0.5, label="Guidance Scale (CFG)")
                seed_in = gr.Number(value=42, precision=0, label="Seed")
                auto_trig_in = gr.Checkbox(value=True, label="Auto-prepend 'in sks_mithila style'")
            run_btn = gr.Button("Generate Artwork", variant="primary")
        with gr.Column(scale=3):
            img_out = gr.Image(label="Generated Image (1024×1024)", type="pil")

    gr.Examples(
        examples=EXAMPLE_PROMPTS,
        inputs=[prompt_in, lora_scale_in, steps_in, cfg_in, seed_in],
    )
    run_btn.click(
        fn=generate_image,
        inputs=[prompt_in, lora_scale_in, steps_in, cfg_in, seed_in, auto_trig_in],
        outputs=img_out,
    )

if __name__ == "__main__":
    demo.launch()
