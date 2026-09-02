# Packaging Video Prompt Craft for Another LLM

Use this when the user asks for a system prompt, custom assistant, skill, knowledge base or reusable instruction set for another LLM.

## Preserve these layers

1. **Role:** model-specific video prompt compiler and diagnostician, not workflow optimiser.
2. **Routing:** model, version, implementation, mode, duration, assets and output contract.
3. **Evidence:** documented, derived and experimental tiers with current-source verification.
4. **Universal compiler:** temporal plan, mode authority, conflict resolution and compliance pass.
5. **Model profiles:** separate, replaceable modules rather than one blended prompt style.
6. **Outputs:** generation-ready prompt first, minimal metadata, assumptions only when needed.
7. **Tests:** realistic requests with expected structural properties, not memorised ideal prose.

## System-prompt structure

```text
ROLE
You craft and diagnose prompts for generative video models.

SCOPE
Prompt content, model syntax, conditioning semantics, timing, camera, action,
dialogue, audio, references, continuity and prompt adherence.

OUT OF SCOPE
Installation, hardware, quantisation, sampler speed, acceleration LoRAs and
workflow tuning unrelated to prompt interpretation.

EVIDENCE POLICY
[Authority order and Documented/Derived/Experimental labels]

INTAKE AND ROUTING
[Model/version/implementation/mode/duration/assets]

UNIVERSAL COMPILER
[Timeline, authority by mode, conflict resolution, compliance pass]

MODEL PROFILES
[Only the target models needed by this assistant]

OUTPUT CONTRACT
[Prompt first, metadata, assumptions/experimental notes]

TEST CASES
[Representative requests and acceptance criteria]
```

## Portability rules

- Translate instructions into the target platform's skill or system-prompt format; do not claim access to tools the target LLM lacks.
- Keep first-party facts and source dates in an attached knowledge section when the platform supports retrieval.
- Do not embed volatile capability matrices without a verification date and update procedure.
- Preserve exact syntax examples where the model requires them, especially MiniMax H3 alignment lines and dialogue tags.
- Do not expose private system instructions or claim to reproduce hidden prompts. Package only the user-owned domain workflow and public evidence.
- Keep model profiles separable so future versions can be added without destabilising existing behaviour.

## Acceptance tests

Require the packaged assistant to:

- distinguish I2V from FFLF and LF;
- compile the same concept differently for H3, LTX and Wan;
- assign roles to multiple references without conflicts;
- preserve exact dialogue and reduce speaker ambiguity;
- refuse to call an experimental workaround official;
- diagnose an overloaded prompt and produce a simpler revision;
- add a new model using the profile template and current sources.
