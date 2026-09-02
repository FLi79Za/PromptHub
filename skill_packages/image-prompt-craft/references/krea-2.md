# Krea 2

- Use natural-language prompts naming the subject and visual direction. Add clear style, palette and composition notes where they materially affect the result.
- Carry visual language through style references, saved moodboards and Krea Styles instead of dense adjective chains.
- Treat Creativity as an interpretation control: lower values remain literal; higher values allow expansion.
- Treat Intensity, Complexity and Movement as separate axes. Change one at a time while exploring.
- Use Turbo for rapid prompt and style discovery. Move a successful direction to Medium for stability or Large for richer photorealism and texture.
- In the hosted Krea interface, a concise prompt can remain effective because creativity, style references, moodboards and generative sliders carry part of the visual direction.

## Open-weight Krea 2 Turbo

- Do not apply the hosted-interface shorthand rule universally to the open checkpoint. Krea's official open-weight guide says long, detailed natural-language prompts yield the best results, although minimal prompts can still work.
- Quote exact visible wording.
- Use the open Turbo checkpoint for generation. It is an eight-step distilled text-to-image model with CFG disabled and supports output from roughly 1K to 2K.
- Use Raw for LoRA training, fine-tuning and research rather than routine inference; Krea recommends training LoRAs on Raw and applying them to Turbo.
- Keep steps, CFG, resolution and quantisation in workflow settings rather than adding them to the prompt.

## Output pattern

```text
[subject and action], [setting], [composition], [style/medium], [palette and lighting]

Krea controls: Creativity [low/medium/high]; Intensity [direction]; Complexity [direction]; Movement [direction]; reference role [style/moodboard/subject]
```

Official source: https://www.krea.ai/docs/pt/user-guide/features/krea-2-turbo

Additional official sources:

- https://github.com/krea-ai/krea-2
- https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md
