"""Gradio demo for HF Spaces (ZeroGPU or CPU-offload). PLAN.md step 5."""

import gradio as gr

demo = gr.Interface(fn=lambda p: None, inputs=gr.Textbox(label="Prompt"), outputs=gr.Image(), title="SDXL LoRA demo (WIP)")

if __name__ == "__main__":
    demo.launch()
