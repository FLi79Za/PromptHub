# Z-Image

- Use explicit, visually grounded descriptions of count, pose, layout, lighting and visible wording.
- Treat Z-Image Turbo and the foundation model separately. Turbo is distilled, uses eight NFEs and does not use CFG; do not attach generic CFG or negative-prompt recipes to it.
- Use the official Prompt Enhancer when world knowledge or richer interpretation helps. Prefer direct prompting for strict composition or exact text.
- Exploit stated strengths in photorealism and English/Chinese text, while still quoting exact wording and defining placement.
- Use negative prompting only for variants that officially support it, such as the foundation model.
- Route native editing to Z-Image-Edit or an editing-capable variant. Do not present a frontend's Turbo inpaint workaround as an official native capability.

## Local note

The official project describes Z-Image Turbo as a 6B model that fits within 16 GB VRAM. It is a priority local text-to-image option for an RTX 4070 Ti SUPER.

Official source: https://github.com/Tongyi-MAI/Z-Image

