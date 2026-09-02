# LTX-2.3 and LTX-2.5 Prompt Profile

Verified: 2026-08-23

## Contents

- Shared grammar
- Version gate
- T2V
- I2V and FFLF
- A2V / IA2V
- Multi-shot
- Retake, extend and reframe
- Dub-It and controlled video modes
- Cautions

## Shared grammar

Use natural-language prose, not MiniMax field labels. Include the shot, scene, chronological action, character definition where needed, camera motion relative to the subject, and audio. Use present tense, physical emotion cues and quoted dialogue. Keep lighting coherent and the scene focused.

For a continuous take, use one flowing paragraph of roughly 4–8 useful sentences as a starting point, not a quota. For dialogue, multiple beats or precise timing, screenplay-style formatting is officially accepted when it improves clarity.

## Version gate

LTX-2.3 and LTX-2.5 share the core grammar, but native multi-shot prompting is a documented LTX-2.5 feature. Do not silently apply 2.5 assumptions to 2.3.

As of the verification date, the LTX API lists T2V, I2V and A2V for both 2.5 variants. Retake, extend and reframe are listed for LTX-2.3 Pro, not LTX-2.5. Local open-source workflows and IC-LoRAs can expose additional controls; identify those by exact workflow rather than claiming generic version support.

## T2V

Establish all important visible and audible facts. A reliable order is:

`main action -> motion mechanics -> subject and environment -> camera and lighting -> audio -> end state`

LTX's 2.3 adherence guidance recommends concise, specific prompts, with the main action early and complexity matched to duration. Treat any numeric word ceiling as guidance, not a hard model limit.

## I2V and FFLF

For I2V, the image already defines appearance and composition. Focus on what happens next: movement, expression change, camera behaviour, scene evolution and audio. Mention visible details only to identify what must remain stable.

For a supplied last frame, describe the motion between endpoints and avoid camera instructions that make the final geometry impossible. A last frame fixes the endpoint and therefore requires a fixed duration in the current 2.5 API. Prefer a continuous take from an opening image unless the user intentionally wants the shot to cut away.

## A2V / IA2V

Audio establishes temporal structure. Describe the intended subjects, setting, performance, camera coverage and visible interpretation of speech, music or ambience. Do not rewrite or re-time the supplied audio in the text prompt.

The API calls this Audio-to-Video and can optionally accept a first image, last image and camera motion. Local workflows may call related conditioning IA2V. Confirm whether the implementation expects an image, audio, both, or additional identity/control inputs.

## Multi-shot

Use only for LTX-2.5 unless another exact implementation documents it. Write 2–4 connected shots as chronological prose. At every cut:

1. name the transition in natural language;
2. establish the new framing and angle;
3. re-identify recurring subjects with the same descriptors;
4. state what audio continues, changes or drops.

Give each shot a distinct editorial purpose. Avoid a bare numbered shot list without prose transitions. Stay single-shot for unbroken motion, intimate performance, or dialogue that needs stable lip-sync in one framing.

## Retake, extend and reframe

For LTX-2.3 Pro retake, describe only the replacement inside the selected time range and mode: video, audio, or both. State boundary continuity and leave preserved material alone.

For extend, describe what happens in the new beginning or ending segment. Begin from the visible and audible boundary state, continue motion and sound, and end on a new observable state. Avoid recapping the whole source clip.

Reframe primarily changes canvas geometry. If a prompt field is available in the implementation, describe only what newly generated peripheral regions should contain while preserving the original action, subject scale and visual continuity.

## Dub-It and controlled video modes

Dub-It is an LTX-2.5 IC-LoRA video-to-video speech-replacement workflow. Describe the replacement spoken content and speaker performance while preserving the source video unless the user requests another change. Keep exact wording quoted and specify language where needed.

For depth, pose, motion, identity or other IC-LoRA control, let the control input own structure. Use text for appearance, setting, intended action, audio and exceptions that the control signal does not provide. Do not duplicate or contradict the control trajectory.

## Cautions

- Do not copy MiniMax XML-like dialogue tags into LTX prompts.
- Do not force multi-shot because 2.5 supports it.
- Avoid generic negative-prompt lists. Use a supported negative field only for specific exclusions.
- Prompt enhancers can add useful detail, but review them for altered dialogue, identity, geography, endpoints and unnecessary camera moves.
- Exact typography and chaotic physics remain unreliable; do not promise them.
