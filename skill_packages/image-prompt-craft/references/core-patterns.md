# Core Prompt Patterns

Use these as content checklists, not mandatory syntax.

## Text to image

```text
PURPOSE: photo, illustration, poster, product shot or concept frame
SUBJECT: identity-defining traits, materials, clothing and count
ACTION: pose, gesture or interaction
SCENE: location, era, weather and background
COMPOSITION: shot size, viewpoint, placement, perspective and aspect ratio
LIGHT/COLOUR: source, direction, hardness and palette
STYLE/MEDIUM: photographic treatment or physical/digital medium
TEXT: exact wording, typography, placement and occurrence count
CONSTRAINTS: required geometry, visibility, count and exclusions expressed positively
```

Lead with the semantic core. Add detail only when it changes the desired image. Prefer concrete visual relationships to adjective chains.

## Editing and compositing

```text
CHANGE: [one precise modification and its location]
PRESERVE: [identity, pose, framing, geometry, background, text, lighting]
MATCH: [perspective, scale, colour temperature, grain, shadows, reflections]
OUTPUT: [aspect ratio, transparency or finish when supported]
```

Assign every input image a role: base scene, identity, object, wardrobe, pose, layout or style. Specify where transplanted elements belong. Do not rely on attachment order alone.

## Exact text

- Quote the required wording verbatim.
- State where it appears, its hierarchy and visual treatment.
- State whether it appears once only.
- Keep copy short where possible.
- Separate visible wording from descriptive prose.

## Photorealism

Describe plausible capture conditions, materials and imperfections rather than relying on `photorealistic` alone. Useful details include natural skin texture, fabric wear, contact shadows, practical light sources, restrained grading and ordinary environmental detail.

## Iteration

Keep successful invariants explicit. Request one major change per pass for identity-sensitive, product, architectural and typographic work. Diagnose failures by category: missing subject fact, ambiguous spatial relation, conflicting style, overloaded composition, weak invariant or unsupported model control.

