# MiniMax H3 Prompt Profile

Verified: 2026-08-23

## Contents

- Capability map
- Base format
- T2VA
- I2VA
- FL2VA
- L2VA
- Ref2VA
- Dialogue and audio
- Cautions

## Capability map

MiniMax documents T2VA, I2VA, FL2VA, L2VA and full-reference Ref2VA. The API groups first-frame, last-frame and first-plus-last-frame under first/last-frame I2V. Ref2VA can combine reference images, videos and audio and assign them to identity, style, motion, camera, voice or editing rhythm.

ComfyUI also exposes guide anchoring at arbitrary frames and masked regeneration/extension. Treat those as implementation-specific conditioning modes and keep the prompt focused on the intended timeline or edited region.

## Base format

Use MiniMax's documented three fields in this order:

```text
integrated_multimodal_description: [Shot 1] ...

overall_soundscape: ...

non_diegetic_music: ...
```

The main field contains visuals, action, shot changes, speakers, dialogue, singing and synchronised diegetic events in playback order. The soundscape is a short global summary of ambience, physical sounds and non-verbal human sound. Do not repeat dialogue or diegetic music there. The music field contains only music the characters cannot hear; use `N/A` when none is wanted.

Start `[Shot 1]` without a timestamp. For later shots use sequential numbers and a strictly increasing cut time inside the duration:

```text
[Shot 2] At 00:04.500, the camera cuts to ...
```

Use natural camera sentences. Include motion type plus meaningful speed or amplitude, not a tag pile.

## T2VA

Begin directly with the three fields. Establish style, initial composition, subjects, setting and action from text. Write timed shots only when the user wants multiple shots and the duration supports them.

## I2VA

Start with the documented alignment line, followed by a blank line:

```text
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.
```

Then use the three fields. Anchor the opening identity, clothing, composition, scene and critical objects from `<Picture 1>`, then describe action onset, development and result. Do not re-invent the source frame.

## FL2VA

Start with:

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.
```

Replace `N` and `S.SS` with the actual final shot and effective duration. Prefer one continuous shot unless multiple shots are explicitly required. Describe observable intermediate changes that progressively close the difference between the two frames. Land on the supplied final pose, spacing, camera, lighting and composition.

## L2VA

Start with:

```text
How the reference pictures align with the target video — <Picture 1> (from [Shot N]) aligns with the S.SS-second mark of the target video.
```

Infer a compatible earlier state, then describe a path that converges on the last-frame image. Do not treat the last frame as the opening frame.

## Ref2VA

Use the documented six-section rewrite format in this order:

```text
subject_definitions:
...

summary:
...

retention_analysis:
...

detailed_description:
[Shot 1] ...

overall_soundscape:
...

non_diegetic_music:
...
```

Use `<Subject N>` for reusable visible content, `<Picture N>` for a concrete frame or composition anchor, `<Video N>` for whole-video structure, camera, rhythm, continuation or editing source, and `<Audio N>` for copied or referenced audio. Keep every label stable across all sections.

Define what each source contributes. If a picture only supplies a person's identity, cite it inside that subject definition rather than pretending the whole picture is a target frame. In `retention_analysis`, state what is fully preserved, partly preserved, transferred, adapted, or intentionally ignored. In `detailed_description`, cite the relevant subject and asset tags where they take effect.

Do not use the six-section format for base T2VA/I2VA/FL2VA/L2VA unless the selected implementation explicitly routes through full-reference rewriting.

## Dialogue and audio

Assign stable `(S1)`, `(S2)` IDs only to vocalising subjects. Put identity, action and delivery outside the dialogue tag; place only the language and exact words inside:

```text
The tired man in the driver's seat (S1) looks at Sam and says in a dry voice: <d>[English] And this affects our motel situation how?</d>
```

Keep an ID attached to the same voice across shots. A compound ID such as `(S1,S2)` is for genuinely simultaneous speech. For voiceover, use `says in an off-screen voiceover` and explicitly keep the on-screen character's lips closed. Use `<scenetrans>` only for dialogue or lyrics that cross a cut, and `<cutoff>` only when the line is intentionally truncated at the end.

Speaker tags reduce ambiguity but do not guarantee correct attribution. For two-person exchanges, keep the speaker clause adjacent to the line, describe the listener as silent, and split lines across shots or clips if swapping persists.

Visible on-screen text should be inside English double quotation marks. Do not promise exact spelling across frames.

## Cautions

- MiniMax's official structured format can be long, but length is not a target. Remove redundant detail.
- Do not treat a claimed fixed word-per-second or one-speaker-per-shot limit as official. Use them only as cautious, testable heuristics.
- Camera tags such as `[pan]`, `[zoom]` and `[static]` are documented by the API, while the open-weights writing guide prefers natural camera sentences. Follow the selected implementation's expected format.
- H3-Context-IR and other prompt enhancers may help structure inputs but must not change exact dialogue, reference roles, endpoints or critical actions without review.
