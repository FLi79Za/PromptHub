---
name: craft-video-prompts
description: Craft, rewrite, diagnose, compare, and package efficient prompts for generative video models and conditioning modes. Use for MiniMax H3, LTX-2.3, LTX-2.5, Wan 2.2, or another video model; for T2V, I2V, first-frame, first-and-last-frame, last-frame, reference-to-video, image/audio-to-video, speech-to-video, video-to-video, retake, extend, reframe, character animation/replacement, inpainting, multi-shot, or runtime-specific modes; for prompt-adherence, dialogue-speaker, camera, action, continuity, timing, reference-assignment, audio, or negative-prompt problems; and for converting this expertise into a system prompt, assistant specification, knowledge base, or skill for another LLM. Exclude workflow installation, hardware, quantisation, sampling speed, LoRAs used only for acceleration, and unrelated node-graph optimisation.
---

# Craft Video Prompts

Produce the smallest prompt that fully communicates the requested audiovisual event to the selected model and mode. Preserve the user's intent, supplied references, exact dialogue, and fixed start/end states.

## Route the request

1. Identify the model, version, implementation, mode, duration, input assets, desired output, and whether the clip is a single take or multi-shot.
2. Ask only for missing facts that materially change the prompt. If the user supplied an image or existing prompt, inspect it before rewriting.
3. Distinguish the model's native capability from a wrapper, custom node, API endpoint, or community workaround. A shared mode name does not imply identical conditioning.
4. Read the matching model reference completely:
   - MiniMax H3: [references/minimax-h3.md](references/minimax-h3.md)
   - LTX-2.3 or LTX-2.5: [references/ltx-2x.md](references/ltx-2x.md)
   - Wan 2.2: [references/wan-2-2.md](references/wan-2-2.md)
   - Unknown or new model: [references/model-profile-template.md](references/model-profile-template.md)
5. Read [references/common-prompt-craft.md](references/common-prompt-craft.md) for action, continuity, dialogue, camera, timing, reference, and failure-analysis rules.
6. When creating a system prompt, skill, assistant profile, or knowledge base for another LLM, also read [references/portable-llm-prompts.md](references/portable-llm-prompts.md).

## Apply the evidence policy

Use this authority order:

1. Current official model documentation, model card, repository, or vendor prompt guide.
2. Current official integration documentation that exposes the model's actual conditioning format.
3. User-supplied tests and documents, when their implementation matches the request.
4. Reputable third-party tests or repeated community observations.
5. Unverified anecdotes.

Tag non-obvious advice internally as:

- **Documented**: directly supported by a current first-party source.
- **Derived**: a conservative implication of documented behaviour or basic cinematography.
- **Experimental**: implementation-specific, weakly evidenced, or inconsistent across reports.

Do not present Derived or Experimental advice as a model rule. If current facts or a claimed supported mode matter, verify them online. Record source URLs and a verification date in any reusable profile. Use [references/sources.md](references/sources.md) as the starting source map.

## Compile the prompt

### 1. Build the temporal plan

Express the clip as a causal sequence:

`initial state -> initiating action -> intermediate change -> reaction/consequence -> end state`

Budget only as many independent beats, speakers, subjects, cuts, camera moves, and effects as the duration can plausibly hold. Split overloaded concepts or explicitly warn about the trade-off.

### 2. Allocate authority by mode

- **T2V:** Text must establish every important visible and audible fact.
- **I2V / first frame:** The image owns initial identity, wardrobe, setting, composition, and lighting. Prompt mainly what changes after frame zero.
- **FFLF / interpolation:** The images own both endpoints. Prompt only a physically and visually plausible path between them.
- **LF / end-frame:** The image owns the final state. Infer a compatible opening and describe convergence to the supplied frame.
- **R2V / multimodal reference:** Assign one explicit job to each asset: identity, wardrobe, location, style, motion, camera, voice, sound, rhythm, or edit structure. Resolve conflicts and state which source wins.
- **IA2V / A2V / S2V:** Audio owns timing and vocal or rhythmic events. Prompt the intended visible interpretation, performance, framing, environment, and any motion not already determined by audio or pose control.
- **V2V / retake / inpaint / replace:** The source video owns all unchanged regions and continuity. Describe only the selected change, its time or mask scope, and how it rejoins preserved material.
- **Extend / continue:** The source ending or beginning owns continuity. Describe the new segment from the boundary outward without restating or changing locked facts.
- **Animate / motion transfer:** Driving media owns motion and timing; the character image owns identity. Prompt appearance-preserving performance, environment and intended exceptions, not duplicate choreography.

### 3. Resolve competing instructions

Prioritise in this order unless the user says otherwise:

1. supplied endpoint frames and masks;
2. explicitly assigned reference roles;
3. exact dialogue or audio timing;
4. subject identity and spatial continuity;
5. primary action and consequence;
6. camera behaviour;
7. lighting, style and secondary detail.

Remove contradictions rather than repeating them as negatives. Never invent a reference tag or field syntax that the selected implementation does not expose.

### 4. Use model-native syntax

Compile into the exact structure in the selected model reference. Do not flatten MiniMax H3's documented structured format into an LTX paragraph, or impose MiniMax tags on Wan or LTX.

### 5. Run the compliance pass

Check that:

- every named subject has one stable description and unambiguous pronouns;
- actions are chronological and physically connected;
- hands, props, weapons, clothing and screen direction do not change without explanation;
- subject direction and camera direction are distinct;
- only one camera logic applies at any instant;
- the requested end state is observable;
- dialogue belongs to the correct speaker and fits the available time;
- audio instructions do not duplicate or contradict supplied audio;
- references have explicit roles and no unresolved conflicts;
- all model-specific tags, timestamps, fields and transitions are valid;
- unsupported guarantees such as perfect text, physics, lip-sync or identity are not claimed.

## Diagnose weak generations

When the user reports a failure, identify the smallest likely cause before rewriting:

- wrong model/version/mode profile;
- prompt describes state instead of change;
- too many beats for the duration;
- ambiguous actor, pronoun, prop, direction or reference role;
- conflicting camera or endpoint geometry;
- dialogue attribution or word-budget overload;
- prompt syntax not recognised by that implementation;
- prompt enhancer changed critical wording;
- the requested behaviour exceeds documented capability.

Return one conservative revision first. Offer an experimental variant only when it tests a clear hypothesis, and label it.

## Default response contract

Lead with the generation-ready prompt. Then include only useful metadata:

`Model | Implementation | Mode | Duration | Structure`

Add a brief **Assumptions** or **Experimental note** only when needed. Do not bury the prompt beneath an explanation. When comparing models, provide one separately compiled prompt per model rather than a universal prompt with cosmetic edits.

For prompt critique, preserve the user's wording where it already works and explain only the changes that materially affect adherence.
