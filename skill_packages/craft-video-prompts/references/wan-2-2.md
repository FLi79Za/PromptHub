# Wan 2.2 Prompt Profile

Verified: 2026-08-23

## Contents

- Capability map
- Core grammar
- T2V and TI2V
- I2V
- FFLF and LF
- S2V and pose+audio
- Animate and Replace
- Prompt extension and negatives
- Runtime-dependent modes

## Capability map

The official Wan 2.2 repository documents T2V-A14B, I2V-A14B, unified TI2V-5B, S2V-14B, pose-plus-audio S2V, and Animate-14B in animation and replacement modes. Alibaba's Wan 2.2 service also documents first-and-last-frame generation. Do not assume every checkpoint or local wrapper supports every mode.

VACE is officially a Wan 2.1 family. A wrapper may combine or adapt it with Wan 2.2 weights, but label that route Experimental or implementation-specific until an official Wan 2.2 VACE release or documentation exists.

## Core grammar

Use the official conservative formula:

`entity + scene + motion`

Expand only as needed:

`entity detail + scene detail + motion detail + aesthetic control + style`

Motion includes the subject, environment and camera. Use concrete visible actions, amplitude and speed when relevant. Put the primary event early, then specify shot size, camera relationship, lighting and style. Avoid keyword soup.

## T2V and TI2V

In T2V, define all important subject, environment, action and camera information. Keep the event chronological and give the clip one clear visual centre.

The TI2V-5B checkpoint uses the same model for T2V or I2V depending on whether an image is supplied. Compile the prompt according to the actual input, not the checkpoint name.

## I2V

The image owns the starting appearance, composition and aspect relationship. Focus the prompt on the motion that begins from it, environmental movement, camera behaviour and final state. Wan can generate from an image without a manual prompt when prompt extension is used, but a reviewed motion-focused prompt gives the user clearer control.

## FFLF and LF

Wan 2.2 first-and-last-frame generation is documented in Alibaba's service. Describe a smooth, plausible bridge between the images. Preserve endpoint identity, layout, camera geometry and light unless the endpoint itself changes them.

Treat last-frame-only generation as runtime-dependent unless the exact checkpoint or wrapper documents it. If available, use generic LF semantics: infer a compatible opening and converge on the supplied final image. Label reverse-generation or other workarounds Experimental.

## S2V and pose+audio

Wan2.2-S2V uses a reference image plus audio and optional prompt; audio length controls the generated duration unless the implementation overrides it. Prompt the visible subject, environment, performance and camera. Do not transcribe supplied audio as though it must be generated again.

When a pose video is supplied, pose and audio own body motion and timing. Use the prompt for subject appearance, setting, performance intent, camera, lighting and any deliberate deviation that the implementation permits.

## Animate and Replace

Animation mode transfers motion from a driving video to a character image. Keep the prompt concise and compatible with the driving performance. Do not redescribe every gesture unless the user wants a deliberate change.

Replacement mode swaps the source-video character with the character from the image while retaining source background, motion and timing. Describe the desired character integration, appearance-critical details and any intended relighting or style exception. Do not ask the prompt to replace aspects that the mask/background controls preserve.

## Prompt extension and negatives

The official repository recommends optional Qwen-based prompt extension for T2V and vision-language prompt extension for I2V. Treat the extended text as a draft. Verify that it has not added new subjects, changed the action, contradicted the image, introduced camera moves, or diluted the requested ending.

Use a negative prompt only when the selected implementation exposes one. Keep it specific. Do not import an unrelated stock negative list from image generation.

## Runtime-dependent modes

Custom ComfyUI/Wan wrappers may expose FLF, LF, loop, clip join, VACE editing, control video, trajectory, depth, masks or continuation. Before prompting:

1. identify the checkpoint and wrapper;
2. identify which input owns motion, structure, identity, start and end states;
3. confirm the wrapper's required tag or prompt syntax from its documentation;
4. apply the generic mode rules in SKILL.md;
5. label unverified behavioural advice Experimental.
