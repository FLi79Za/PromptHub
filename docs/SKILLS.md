# PromptHub Agent Skills

PromptHub now keeps portable Agent Skills as source packages and stores only catalogue/runtime metadata in SQLite. Imported packages live below `skill_packages/<skill-name>/`; `SKILL.md`, references, examples, schemas, and scripts are preserved. `.prompthub` data is reserved for local adapters and is excluded from portable exports by default.

## Import lifecycle

`inspect_skill` validates the package boundary, parses lightweight front matter, inventories resources, hashes the source tree, and statically reports capability words. ZIP extraction rejects absolute paths and `..` traversal, enforces file/count limits, and never executes scripts. `import_skill` is idempotent for identical hashes and refuses to overwrite a locally modified catalogue record.

## Runtime lifecycle

`run_skill` resolves a skill by id or name, loads `SKILL.md`, deterministically selects matching `references/` files from the requested target/model and request, loads only those resources, applies local adapter metadata, reports capability substitutions/unavailable features, and calls the existing Ollama generator. JSON schemas in `schemas/*.json` are passed as Ollama `format` and validated on return. `resolve_skill_dependencies` provides ordered dependency resolution with missing/circular dependency errors; dependency packages remain independent. Each result stores the skill, model, parameters, selected resources, capability report, and structured result; a non-sensitive trace is recorded in `skill_execution_traces`.

The deterministic router intentionally ignores standalone numeric tokens, so `Flux 2 Klein` does not accidentally load every `*-2.md` reference. It is a replaceable seam for later semantic retrieval.

## API and clients

The authenticated Integration API exposes `/skills`, `/skills/inspect`, `/skills/import`, `/skills/<id>`, `/skills/<id>/run`, and `/skills/<id>/export`. The Codex PromptHub MCP client/server calls those endpoints; it does not open the database or implement a second runtime. The web UI exposes Skills alongside the existing AI Action UI.

## Capability registry and adapters

Runtime capability reports classify local mappings as `PRESERVED`, `SUBSTITUTED`, `DEGRADED`, or `UNAVAILABLE`. The current app maps structured output to Ollama and image generation to the generic future ComfyUI provider slot; it does not execute imported scripts or claim that ComfyUI rendering occurred. Per-model instruction overrides belong in `runtime_config_json`, not in the canonical Skill package.

## Worked examples

Import Image Prompt Craft, choose `qwen3.5:9b`, and run with `parameters={"target_model":"Flux 2 Klein"}` and request `Create a photorealistic cyberpunk detective portrait in heavy rain.`. The runtime loads the Skill plus `references/flux-2.md` and returns the specialised prompt.

Import Video Prompt Craft, choose `qwen3.5:9b`, and run with `parameters={"target_model":"MiniMax H3"}` and request `Create a 12-second MiniMax H3 cinematic fight sequence in a work cafeteria.`. The runtime loads `references/minimax-h3.md`; unrelated LTX/WAN references remain out of context.

## Next steps

The next safe increment is a preview-first standalone importer using this same engine for drag/drop folders and ZIPs, followed by richer front-matter dependency declarations, model-aware semantic retrieval, explicit trusted capability providers (including ComfyUI), and a full compare/merge/export UI. Automatic skill recommendation and multi-skill workflows should build on `search_skills`, dependency resolution, and `run_skill` rather than duplicate specialist instructions.

## Phase 2: orchestration and providers

Dependencies are independent Skill records. Portable front matter may declare `dependencies`; where older Skills only name `$other-skill` in their instructions, PromptHub stores an explicitly marked inferred dependency in catalogue metadata without changing `SKILL.md`. Required missing dependencies block execution; optional missing dependencies are skipped; cycles are rejected.

The runtime creates a root trace before recursively running dependencies. Each child has a `parent_execution_id`, records only selected resources/capabilities/results (never hidden reasoning), validates a declared JSON schema, and returns its structured result to the parent as an operational record. The Director example therefore executes Video Prompt Craft separately rather than copying its rules into the Director context.

`FULLY_OPERATIONAL`, `DEGRADED`, and `BLOCKED` runtime states are calculated from package presence, required dependencies, and provider readiness. Resource selection remains deterministic first; the local adapter can set a strict `context_budget_chars`, and a request that cannot fit fails explicitly instead of silently truncating `SKILL.md`.

Capability providers are privileged components registered in `skill_capability_providers`. A provider requires an enabled configuration, the Skill declaring the abstract capability, and a `trusted` Skill state. The included ComfyUI provider accepts only PromptHub-configured API-format workflows and node-input mappings, checks `/system_stats`, queues through `/prompt`, and returns the normalised ComfyUI prompt id. It does not alter ComfyUI workflows and it does not make Image Prompt Craft render automatically.

Safe maintenance is provided by `compare_skill` and `update_skill`: package hashes and file inventories classify identical/newer/older/conflicting input, show added/removed/modified files, refuse local portable-source conflicts, and preserve local adapters during a safe source update. These same operations are exposed to the MCP client; no Codex-specific updater exists.

## Skill operations and prompt derivations

The shared execution contract now accepts `operation`, typed `inputs[]`, `target`, the local Ollama `model`, `parameters`, and an optional `source_prompt_id`. `CREATE` requires a text input with role `brief`; `TRANSFORM`, `CONVERT`, `REFINE`, and `DIAGNOSE` require role `source`. The legacy `request` field remains compatible and maps to a `CREATE` brief. Text is the only enabled input type in this phase; image, file, audio, and video inputs return `UNSUPPORTED_SKILL_INPUT_TYPE` instead of being ignored.

Operations communicate task semantics to the selected portable Skill; PromptHub does not contain model-specific prompt-writing rules. Targets are read from PromptHub-local target/adapter metadata when available and otherwise discovered from model-profile resources. Choosing `ideogram_4`, `flux_2`, or `minimax_h3` continues to drive deterministic progressive resource selection, so unrelated profiles remain outside the Ollama context.

The prompt editor's **Apply Skill** action submits the current prompt as a source input, keeps the execution model visually distinct from the output target, and displays an original/result comparison. Running never edits the source. A completed execution can be copied, explicitly used to replace the original, or saved as a new child prompt through `/api/prompts/<id>/skill-derivatives` (authenticated clients use `/api/integration/v1/prompts/<id>/skill-derivations`). `prompt_skill_derivations` records the direct source, derived prompt, Skill/version, operation, target, execution model, execution id, selected-resource metadata, and creation time.

Example conversion request:

```json
{
  "operation": "convert",
  "inputs": [{"type": "text", "role": "source", "content": "A cinematic portrait in neon rain"}],
  "target": "ideogram_4",
  "model": "qwen3.5:9b",
  "source_prompt_id": "PROMPT-SYNC-ID",
  "parameters": {}
}
```

After reviewing the returned `result_text`, save it non-destructively with the returned `execution_id`. The source prompt and any Flux/Ideogram derivatives remain independently editable while retaining the same direct lineage.
