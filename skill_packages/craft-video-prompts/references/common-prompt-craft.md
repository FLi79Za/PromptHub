# Common Video Prompt Craft

## Contents

- Prompt economy
- Temporal and physical logic
- Camera and editing
- References and endpoints
- Dialogue and audio
- Negative instructions
- Action and multi-subject scenes
- Failure analysis

## Prompt economy

Include only details that influence the generated clip. Give each sentence one job. Prefer concrete, observable descriptions over abstract mood claims: show shoulders tightening, a hand trembling, or rain flattening clothing rather than saying merely "tense".

Start with the highest-value information for the selected mode. In T2V this is usually subject, setting and primary action. In I2V it is usually the first motion. In FFLF it is the transition. In reference or edit modes it is the asset assignment or requested delta.

Use stable repeated descriptors for recurring subjects. Avoid synonyms that could be read as new entities. Replace unclear pronouns with names or labels whenever two subjects could fit.

## Temporal and physical logic

Write present-tense events in playback order. Make causality explicit:

- approach before contact;
- contact before reaction;
- release or transfer before a hand performs a new action;
- acceleration before impact;
- impact before debris or recoil;
- camera reveal before newly visible information matters.

Maintain state. If a person holds a newspaper in both hands, describe lowering or transferring it before another two-handed action. If a prop remains important, re-identify it naturally at the moment of use.

For a fixed duration, treat each major action, line, camera move or edit as consuming time. There is no universal seconds-per-beat rule. Simplify until the desired sequence can be performed at a natural speed.

End on an observable state: who is where, what they are doing, what the camera frames, and which important sound remains.

## Camera and editing

Separate subject motion from camera motion. State the camera's relationship to the subject, for example: "The camera tracks parallel at matching speed, keeping her full body in profile."

Use one main camera intention at a time. Combine movements only when their physical relationship is clear. Distinguish:

- pan/tilt: camera position stays fixed while view rotates;
- truck/pedestal/dolly: camera position moves;
- zoom: focal length changes;
- tracking: camera follows a moving subject;
- static/locked: camera and lens remain fixed.

Do not request a cut when a small reframing or camera move is sufficient. For multi-shot prompts, re-establish framing, subject identity, action continuity, geography and audio after every cut.

## References and endpoints

Describe asset roles, not merely asset presence. Example:

`Reference image 1 defines Mara's face and hair only. Reference image 2 defines her wardrobe. Reference video 1 supplies body motion and timing, not identity or location.`

When sources conflict, state precedence. Do not ask the model to preserve two incompatible wardrobes or simultaneously copy two different camera paths.

For I2V, do not spend most of the prompt redescribing the visible frame. Mention visible facts only to disambiguate the subject, preserve a critical element, or explain the change.

For FFLF, compare endpoints before writing. Identify changes in pose, position, prop state, camera geometry, light, environment and subject count. If the endpoints imply an impossible or overloaded transition, warn or propose a simpler bridge.

For LF, describe a plausible prior state and a continuous convergence. The last frame is not automatically the first frame.

## Dialogue and audio

Preserve user-provided dialogue verbatim unless asked to edit it. Attribute the line with the model's documented syntax and place delivery or voice traits outside the quoted spoken text.

Fit dialogue to duration. Leave time for pauses, reactions and physical action. If attribution fails:

1. reduce simultaneous visible speakers;
2. name and position the speaker immediately before the line;
3. state the listener's non-speaking behaviour and closed mouth when useful;
4. separate speakers by shot or generation if the model still swaps lines;
5. treat any stronger workaround as Experimental unless documented.

Do not duplicate supplied audio as newly generated audio. For audio-driven modes, describe how visuals interpret the timeline. For generated audio, distinguish dialogue, diegetic sound, ambience and non-diegetic score according to the model's syntax.

## Negative instructions

Use a negative prompt only when the implementation exposes one and the exclusion is useful. Prefer a short, task-specific list over a generic quality incantation. Do not put desired positive content in the negative field.

When no negative field exists, write the intended positive state. A concise prohibition can help only when necessary and supported; long "do not" lists compete with the actual event.

## Action and multi-subject scenes

Establish geography before choreography: subject labels, screen positions, facing direction, distance and important obstacles. Keep bodies visible when limb interaction matters.

Describe exchanges as intention, response and consequence rather than move names alone. Use realistic momentum and contact points. Preserve the line of action unless a camera crossing is deliberate and clearly motivated.

Reduce simultaneous independent actions. For crowds, establish the overall event, then isolate one local action rather than choreographing every person.

## Failure analysis

Change one variable per diagnostic revision where possible. A useful report contains:

- observed failure;
- likely prompt-level cause;
- conservative correction;
- optional experimental correction;
- what outcome would confirm or reject the hypothesis.

Do not blame the prompt automatically. Conditioning strength, implementation bugs, stale prompts, or model limitations may be responsible, but workflow diagnosis is outside this skill unless the user specifically asks for a boundary check.
