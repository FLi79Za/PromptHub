# Ideogram 4

- Prefer structured JSON when layout, typography, multiple elements or exact colours require strong control. Ideogram states that 4.0 was trained on structured JSON captions.
- Keep canvas properties separate from individual elements. Give each element content, location, visual attributes and relationships.
- Put exact visible wording in a dedicated text element. Define placement, hierarchy, type character and contrast.
- Use supported hex-colour steering for brand work. Official guidance allows up to 16 colours and up to five colours per element.
- Avoid contradictions between positive requirements and exclusions.
- Use Magic Prompt only when extra interpretation is wanted. Preserve manual structure when compliance matters more than surprise.

## Compact structure

```json
{
  "scene": "...",
  "style": "...",
  "composition": {"aspect_ratio": "...", "viewpoint": "...", "layout": "..."},
  "elements": [
    {"type": "subject", "description": "...", "position": "...", "colours": ["#..."]},
    {"type": "text", "content": "EXACT WORDING", "position": "...", "typography": "..."}
  ],
  "lighting": "...",
  "constraints": ["..."]
}
```

Official sources:

- https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide
- https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/4.-json-prompting-ideogram-4.0

## Strict structured-caption renderer profile

When the user supplies or requests the strict renderer schema documented in the Image Prompt Craft knowledge base, use this contract instead of the generic JSON pattern above:

- Emit exactly one minified JSON object with exactly three top-level keys, in this order: `aspect_ratio`, `high_level_description`, `compositional_deconstruction`.
- `compositional_deconstruction` contains exactly `background` and `elements`.
- Each visual subject is one `obj` element: `{"type":"obj","bbox":[y1,x1,y2,x2],"desc":"..."}`. Use one element for one coherent subject, not separate anatomy elements.
- Use `text` elements only for visible wording: `{"type":"text","bbox":[y1,x1,y2,x2],"text":"...","desc":"..."}`.
- Bboxes use normalised 0–1000 coordinates in `[y1,x1,y2,x2]` order, with top-left origin. Commit to the aspect ratio before choosing them.
- Put the scene shell, architecture, ground and ambient lighting in `background`; put individually placeable subjects and objects in `elements`. Do not double-count.
- Keep `high_level_description` observational and under 50 words. Keep object descriptions concrete, identity-first and under 60 words.
- Do not add unsupported top-level keys, markdown fences, commentary, hedge alternatives or generic placeholder elements. Do not invent text unless the brief requires in-scene text.

This is a user-supplied renderer contract and should not be presented as a universal Ideogram API schema.
