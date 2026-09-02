# Local Models on 16 GB VRAM

Separate model prompt craft from runtime configuration. Never place quantisation, offload, sampler or LoRA guidance inside the image prompt unless the user explicitly asks for workflow settings.

## Priority routing

- FLUX.2 Klein 9B distilled: practical high-quality generation/editing target on the user's 16 GB RTX 4070 Ti SUPER based on their tested local workflow.
- FLUX.2 Klein 4B: prioritise when speed and memory headroom matter.
- Z-Image Turbo: official 6B, eight-NFE text-to-image model designed to fit within 16 GB.
- Krea 2 Turbo FP8: credible experimental 16 GB text-to-image route. Community testing reports the roughly 13 GB FP8 checkpoint on 12 GB and 16 GB cards, but Krea publishes no official minimum VRAM. Raw is for training and is not the routine 16 GB inference choice.
- Qwen-Image-2512 / Qwen-Image-Edit-2511: current open-weight Qwen routes for generation and editing respectively. Full-precision operation exceeds a comfortable 16 GB path; treat quantised/offloaded 16 GB support as community implementation evidence.

## Krea 2 Turbo

- Official: Krea 2 Turbo is a 12B-class, eight-step open-weight text-to-image model, and ComfyUI supports it natively. The official sources do not state a 16 GB minimum.
- Community implementation evidence: a tested ComfyUI guide reports FP8 generation on RTX 3060 12 GB and RTX 4080 16 GB systems. Treat the user's RTX 4070 Ti SUPER 16 GB as a viable experimental target with FP8 and normal ComfyUI offload caveats.
- Keep Krea's official runtime state outside the prompt: Turbo uses eight steps, CFG disabled and output up to 2K.

## Qwen editing

- Route local text-to-image work to Qwen-Image-2512 and local editing or multi-image consistency work to Qwen-Image-Edit-2511.
- Qwen-Image-2.0 is the newer hosted generation-and-editing model, but the official open-weight repository does not provide a local checkpoint. Do not present it as locally downloadable.

- Use direct natural-language changes and explicit preservation instructions.
- Use chained, localised corrections for text and detail errors rather than broad repeated rewrites.
- Assign roles to multiple inputs for person, scene, object, pose or control map.
- Expect runtime and quantisation choices to affect quality and speed; do not misdiagnose those effects as prompt failures without evidence.

Official sources:

- https://github.com/QwenLM/Qwen-Image
- https://huggingface.co/Qwen/Qwen-Image-2512
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- https://github.com/krea-ai/krea-2
- https://www.earngenix.com/workflows/krea-2-comfyui
