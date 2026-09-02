# ComfyUI workflow dispatch

PromptHub can dispatch a saved prompt to private/local ComfyUI instances through reusable Workflow Profiles. This feature is intentionally limited to prompt-to-media generation and history. Sequence planning, continuity, editorial order and approved takes remain ShotBinder responsibilities.

## Servers

Open **Generate → ComfyUI** and add a stable id, display name and base URL. Configure image and video installations separately, for example `image-comfyui` at `http://127.0.0.1:8188` and `video-comfyui` at `http://127.0.0.1:8189`.

PromptHub distinguishes configured, reachable, unreachable and disabled servers. **Test** calls `/system_stats`. URLs must resolve to loopback, link-local or private-network addresses; embedded credentials and URL paths are rejected.

## API workflows and profiles

Import the API-format prompt graph produced by ComfyUI's **Save (API Format)**, not the visual editor workflow. API graphs are non-empty objects whose node ids map to `class_type` and `inputs` objects. Imported JSON is untrusted configuration; no scripts are executed.

The exact API source is hashed and preserved under `provider_workflows/comfyui_profiles/<profile-id>/<sha256>.json`. Execution deep-copies it before injecting values. If that source changes, validation fails instead of silently submitting a different graph. When the API graph is derived from an editable UI workflow, record the UI path/hash, API path/hash and converter identity in profile compatibility metadata.

A profile records kind, model family, mode, target server, source hash/version, semantic inputs, relevant output nodes, defaults and validation. It contains execution logistics, not model prompting expertise.

The importer lists literal node inputs and offers cautious suggestions. Map only fields users should control. A mapping records its semantic role, node id, field, type, label, default, multiplicity and required state. Connections such as `["12", 0]` are not offered as literal inputs. Ambiguous fields remain unmapped until chosen. Output-node selection prevents previews and diagnostics from becoming user-facing results.

Validation catches malformed API JSON, unknown servers, missing nodes, missing fields and missing output nodes. Duplicate a profile to create alternate defaults, acceleration variants or server assignments without changing source code.

## Generation and history

Open a saved prompt and choose **Generate**. Compatible enabled profiles are ranked using the prompt's image/video category and its Skill derivation target, when present. Selection remains explicit. The form is generated from the profile and supports text, numeric, boolean, enum, image, multiple-image, audio and video inputs.

Media is uploaded through ComfyUI's API. Integration clients send `{ "filename": "start.png", "content_base64": "..." }`; arbitrary local paths are rejected. PromptHub records both its source reference and ComfyUI's uploaded input reference.

Each generation stores the prompt revision/content hash, profile version/hash, server, ComfyUI prompt id, supplied inputs and coerced parameters. Status refresh reconciles `/history/<prompt-id>` with `/queue`. Outputs are fetched through `/view` into PromptHub-managed storage, so a remote ComfyUI Windows path is never the sole reference.

States are `preparing`, `queued`, `running`, `completed`, `failed` and `cancelled`. Failures keep a readable message and structured technical detail. **Regenerate** reuses the profile and retained inputs; **New Seed** changes a recorded seed.

## Skill to generation

Skill execution never renders automatically:

1. Save a generic image prompt.
2. Apply Image Prompt Craft with target `flux_2_klein`.
3. Save the derivative.
4. Open it, choose **Generate**, select the suggested Flux profile, and submit.

The derivative retains Skill provenance. Target metadata only ranks workflows; it does not embed Flux or H3 prompting rules in PromptHub.

## Image workflow examples

### Flux 2 Klein T2I

Export a known-good Flux 2 Klein API graph and import it against the image server. Map positive text to `prompt`, sampler noise seed to `seed`, and optional dimensions to `width`/`height`. Mark prompt required. Select only the final `SaveImage` node as an image output. Leave model/sampler defaults in the source unless users need typed custom controls.

A live acceptance run must reach terminal history, store/open the retrieved image, retain the effective seed and leave the canonical source SHA-256 unchanged.

### Krea 2 Turbo

Map the prompt at the workflow's prompt-batch/text source, the primary sampler seed and the latent width/height. Expose primary steps, CFG, sampler and scheduler only when they are intentional controls. Do not mistake face-detail, upscaler or camera-simulation seeds for the primary generation seed. Select only intentional saved base/final outputs.

### Ideogram 4

Map the structured positive caption, seed and the width/height bridge feeding the expanded subgraph. Ideogram 4's installed graph uses a text-free unconditional pass rather than a negative-prompt string. Its Quality/Default/Turbo preset owns the coupled step/sigma settings; expose that preset rather than severing its internal controls.

## Refreshing a derived API version

The UI workflow remains the editable source of truth. A safe future **Refresh API Version** operation should:

1. hash the current UI source and retain the previous UI/API pair;
2. compile through the target ComfyUI frontend's `loadGraphData()` and `graphToPrompt()` implementation;
3. write a new API artefact atomically without overwriting the UI source;
4. compare API node classes, mapped node ids/fields and declared output nodes with the active profile;
5. stop and flag broken or type-changed mappings for review;
6. validate the candidate against the target `/object_info` and then perform a reviewed low-cost runtime test;
7. create a new profile version/hash only after validation succeeds.

PromptHub does not currently auto-refresh profiles. This deliberately prevents an editable graph change from silently altering automation behaviour.

## MiniMax H3 I2V example

Export the known-good H3 I2V API graph and target the video server. Map primary text to `prompt`, `LoadImage.image` to required `start_image`, and the sampler seed to `seed`. Expose duration, frames or fps only where the installed node contract confirms literal controls. Select the final video save/combine node as the output.

The start image is uploaded to that specific server and injected into a per-job copy. Never assume Image and Video ComfyUI share custom nodes, models, ports or files.

## Integration API

The authenticated `/api/integration/v1` surface provides:

- `GET|POST /comfyui/servers` and `POST /comfyui/servers/{id}/health`
- `POST /comfyui/workflows/inspect`
- `GET|POST /comfyui/profiles` and `POST /comfyui/profiles/{id}/duplicate`
- `GET /prompts/{sync_id}/compatible-workflows`
- `GET|POST /prompts/{sync_id}/generations`
- `GET /generations/{id}` and `POST /generations/{id}/regenerate`

Codex and other integrations use this API rather than implementing a separate ComfyUI client. Submission is not blindly retried because an uncertain response could create duplicate renders.

## Troubleshooting and acceptance

- **Unreachable:** verify the selected instance, address, port and launcher. One healthy server says nothing about another.
- **Invalidated profile:** re-import or update mappings after reviewing source changes.
- **Missing node/field:** compare the graph with live `/object_info` for the target instance.
- **Upload failed:** confirm the target accepts that media type; remote servers cannot consume a PromptHub path.
- **Execution failed:** inspect retained ComfyUI messages/logs; queue acceptance is not generation success.
- **Missing output:** select the actual final output node.

Real acceptance validates against live `/object_info`, uses a deliberately low-cost representative run, waits for terminal history, opens the retrieved media, and compares the canonical source hash before and after.
