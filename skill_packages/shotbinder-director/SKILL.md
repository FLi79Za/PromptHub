---
name: shotbinder-director
description: Turn a story or creative brief into a reviewable ShotBinder Production Blueprint and dry-run project plan using live ShotBinder capabilities. Use for director-led ShotBinder planning, not direct ComfyUI graph editing or autonomous rendering.
---

# ShotBinder Director

Create a coherent, production-ready plan before creating or rendering anything. Treat ShotBinder as the authority for project persistence, workflow contracts, dependencies, Take Plans, Active/Approved Takes, and queue execution.

## Planning flow

1. Interpret the story, script, scene description, or brief into candidate beats, sequences, creative direction, and a visual bible.
2. Verify `craft-video-prompts` through `list_specialist_dependencies`.
3. Use `$craft-video-prompts` before locking shot boundaries or generation modes. It remains the independently maintained authority for model-specific performance/action research; do not copy its rules into this skill.
4. Convert its review into the versioned `VideoPromptCraftResult`, recording its ID, version, status, recommendations, and source evidence. Codex selects both relevant skills in one task; a Skill does not programmatically execute another Skill.
5. Let the video specialist revise shot boundaries, duration, continuity, performance/action progression, camera strategy, generation mode, guide/reference requirements, temporal prompting, and model-specific prompt format.
6. Query ShotBinder's live semantic capability operations afterwards. Select only supported model/workflow candidates; adapt unavailable specialist recommendations and record the capability constraint.
7. Build and validate a typed `ProductionBlueprint` with explicit prompt roles, media requirements, dependencies, model/workflow recommendations, editorial order, and separate render order.
8. Present a concise production review and run `dry_run_materialization`.
9. Present warnings and conflicts, then use `revise_blueprint` for focused changes rather than regenerating the whole plan.
10. Commit only when the user explicitly asks to create the project, using `commit_blueprint` after review.
11. Inspect unresolved media requirements.
12. Resolve existing user files, ShotBinder assets, upstream Active Takes, optional requirements, or unresolved state without recreating the blueprint.
13. For an unresolved generated first/last frame, guide, character, environment, or prop requirement only, verify and use `$image-prompt-craft`.
14. Convert its output to `ImagePromptCraftResult`, then call `plan_blueprint_media_requirement` before a project exists or `plan_generated_media_requirement` after commit. Do not invoke it for existing user media.
15. Treat image planning and generation as separately reviewable. Attach an accepted asset only with explicit approval via `resolve_media_requirement` / `attach_existing_asset`.
16. Report unresolved requirements or image-backend configuration needs.
17. Stop. Video rendering is a separate explicit action.

## Constraints

- Do not embed or infer raw ComfyUI node IDs, payload paths, or bypass logic.
- Keep prompts explicit: label, role, model family, generation mode, selected state, and stable ID. Generation plans must name the prompt they use.
- Preserve story chronology separately from render order. Dependencies may control readiness but must not silently reorder the story.
- Active Take is operational; Approved Take is editorial. Never replace an Approved Take automatically or use it as a silent substitute for an explicit Specific Take.
- Treat imported workflows as live registry data. Inspect capabilities again rather than hard-coding workflow mechanics.
- Represent unavailable source media as a requirement: user-supplied, ShotBinder asset, upstream Active Take, generated later, optional, or resolved. Do not fabricate paths or generate reference images in this workflow.
- Treat the visual bible as continuity data for characters, locations, props, and visual language. Preserve identity and accepted references; do not redesign them during planning.
- The specialist result contracts are interchange/provenance records, not fallback prompt engines. If a package is unavailable, return an explicit unavailable/awaiting-specialist-review state; never claim it ran or silently substitute Director heuristics.
- Rendering is opt-in. Planning, validation, revisions, dry-run materialisation, and project commit must produce no render submission. Queue preparation and execution require their existing distinct permissions.

## Output

Report the Production Blueprint, capability evidence used, unresolved references or unsupported requirements, and the dry-run materialisation plan. Clearly distinguish recommendations from user-approved project writes.
