# Adding a Video Model Profile

Use this template when a requested model or version has no bundled profile. Research current first-party sources before drafting.

## Required profile fields

```markdown
# [Model and version] Prompt Profile

Verified: YYYY-MM-DD

## Authority
- Official model page:
- Official prompting guide:
- Official repository/model card:
- Official integration docs:
- Runtime-specific docs:

## Native capabilities
- Mode:
- Required inputs:
- Optional inputs:
- What each input controls:

## Prompt grammar
- Required structure or fields:
- Timeline/cut syntax:
- Reference syntax:
- Dialogue/audio syntax:
- Negative-prompt support:
- Prompt limits that affect craft:

## Mode adaptations
- T2V:
- I2V:
- FFLF:
- LF:
- R2V:
- A2V/S2V/IA2V:
- V2V/edit/extend:
- Other:

## Known cautions
- Documented limitations:
- Derived conservative advice:
- Experimental techniques:
```

## Research rules

Prefer the exact version and deployment the user names. Separate platform features from open-weight capabilities. A hosted API, official ComfyUI template and community wrapper may expose different modes for the same base model.

Record direct links and verification date. Quote sparingly; paraphrase guidance. If a source only proves that a mode exists, do not invent its prompt syntax. If no first-party prompt guide exists, use common prompt craft conservatively and label model-specific recommendations Derived.

## Integration rules

Keep the new profile one reference file away from SKILL.md. Add a routing bullet to SKILL.md only when the model is likely to recur. Do not rewrite the universal workflow around one model's quirks.

Test the profile with at least:

1. one straightforward T2V or default-mode request;
2. one conditioned mode using supplied media;
3. one dialogue or audio case if supported;
4. one overloaded or contradictory request that should trigger simplification;
5. one conversion from another model's prompt syntax.
