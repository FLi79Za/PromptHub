# Prompt Hub

Prompt Hub includes reusable AI Actions and a local Knowledge Manager. See [AI Actions and Knowledge](docs/AI_ACTIONS_KNOWLEDGE.md) for collection building, RAW/OPTIMISED/MERGE ingestion, provenance, retrieval testing, audits, rebuilds, and Codex integration.

See [ComfyUI workflow dispatch](docs/COMFYUI_GENERATION.md) for multi-server configuration, reusable semantic Workflow Profiles, media uploads, generation history, API operations, and worked Flux 2 Klein/H3 I2V examples.

A local, self-hosted web application for storing, organising, refining, and reusing AI prompts.

Prompt Hub is built for real-world AI workflows, not toy examples. It supports full-length prompts, instruction-based editing, iterative refinement, visual context, browser capture, local AI integration, and structured organisation.

Everything runs entirely locally.

No cloud services required.

---

## Why Prompt Hub?

As prompt libraries grow, they become messy, duplicated, inconsistent, and difficult to search.

Prompt Hub helps you:

* Organise complex prompts
* Maintain structured experimentation
* Iterate safely with variants
* Store visual and contextual references
* Capture prompts directly from the web
* Run optional local LLM refinement via Ollama
* Build reusable AI Actions from editable system instructions, task templates, models, and local knowledge
* Sync prompt libraries between desktop and iOS

It is designed for serious AI users managing real prompt systems.

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Fli79Za/PromptHub.git
cd <repo-name>

# 2. Create a virtual environment
python -m venv env

# Windows
env\Scripts\activate

# macOS / Linux
# source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

## Docker Compose

From the repository root, run `docker compose up --build -d`, then open
<http://127.0.0.1:5000>. Check startup with `docker compose logs -f prompthub`.
Stop with `docker compose down`. Avoid `docker compose down -v` unless you intend
to delete the stored data. Docker Engine with the Compose v2 plugin (or Docker
Desktop using Linux containers) is required. The image uses Python 3.12.

If port 5000 is already in use, copy `.env.example` to `.env` and set
`PROMPTHUB_PUBLISHED_PORT=5050`, then open <http://127.0.0.1:5050>.
`OLLAMA_HOST` is the other optional setting. Neither variable is required;
defaults are shown in `.env.example`. Local `.env` files are excluded from Git
and the image build. Only application files are copied into the image.

Compose publishes only to the host loopback address. The application listens on
`0.0.0.0` **inside** the container so Docker can forward that loopback port.
The normal `python app.py` launch still binds to loopback. The Integration API
token, Flask session secret, and settings are generated in the persistent
`config` volume; keep that volume private. The five named volumes store:

| Volume | Container path | Contents |
| --- | --- | --- |
| `database` | `/data` | SQLite `prompts.db` and journal files |
| `uploads` | `/app/static/uploads` | Thumbnails and generation media |
| `skills` | `/app/skill_packages` | Imported skill packages |
| `workflows` | `/app/provider_workflows/comfyui_profiles` | Saved workflow profiles |
| `config` | `/config` | Integration API secrets, settings, logs |

On first creation, Docker initialises the skills and workflows volumes from the
files bundled in the image. Existing installations are **not** migrated
automatically. Back up your local database, uploads, skills, workflow profiles
and `%LOCALAPPDATA%\PromptHub` configuration before copying them into the
corresponding volumes. Stop both the source application and the container before
copying SQLite data, or use SQLite's backup API for a consistent source snapshot;
do not copy just a live database while WAL writes are in progress. Imported
skills in an existing database may contain absolute package paths; inspect and
update those records for `/app/skill_packages` when migrating from another host.
Existing named volumes are reused across rebuilds, so updated bundled skills
and profiles are not automatically copied over an existing volume. Review those
updates separately. Keep using the same Compose project name and directory to
reuse the same named volumes.

Ollama and ComfyUI run separately. On Docker Desktop, the default Ollama URL
points to `http://host.docker.internal:11434`. Set `OLLAMA_HOST` in your shell
or a local `.env` file to an address reachable **from the container** if needed.
On Linux, Compose maps `host.docker.internal` to the host gateway, but a service
bound only to host loopback may require its own safe networking configuration.
Configure ComfyUI server URLs in PromptHub to addresses reachable from the
container as well; for a host service on Docker Desktop, for example, use
`http://host.docker.internal:8188`. `localhost` there means the PromptHub container. Neither
external service is installed or published by this Compose file.

Smoke check (adjust the URL if you changed the published port):

```sh
docker compose config
docker compose up --build -d
curl --fail http://127.0.0.1:5000/
docker compose exec prompthub python -c "from app import DB_PATH; print(DB_PATH, DB_PATH.exists())"
docker compose restart prompthub
```

Create a disposable prompt with a thumbnail before restarting and verify both
afterward. To test separately from an existing deployment, use a distinct project
name (`docker compose -p prompthub-smoke ...`) on every command and an unused
`PROMPTHUB_PUBLISHED_PORT`. This gives the smoke deployment separate named volumes.
Stop it with `docker compose -p prompthub-smoke down`; retain the volumes.

## Local Integration API v1

PromptHub includes a local-only, versioned JSON API at `/api/integration/v1` for future ChatGPT app/plugin integration. It supports authenticated prompt search, retrieval, creation, optimistic-concurrency updates, related versions, organisation, dry runs, metadata, and integration change history. Existing UI and browser-extension routes remain unchanged.

Run the database migration once before enabling API clients:

```powershell
.\env\Scripts\python.exe .\tools\migrate_integration_api.py
```

View non-secret status or explicitly retrieve the local bearer token:

```powershell
.\env\Scripts\python.exe .\tools\manage_integration_api.py status
.\env\Scripts\python.exe .\tools\manage_integration_api.py show-token
```

The token, Flask session secret, API settings, and integration logs are stored outside the repository in `%LOCALAPPDATA%\PromptHub` by default. Native launches remain loopback-only. The container explicitly opts into `0.0.0.0` internally, with Compose publishing only on host loopback as described above.

See [Integration API v1 documentation](docs/INTEGRATION_API_V1.md) for configuration, security assumptions, endpoint examples, errors, rollback, and future ChatGPT integration guidance.

---

## 🧠 What Prompt Hub Is Good For

* Image generation prompts (Flux, Stable Diffusion, Midjourney, etc.)
* Instruction-based image editing (Qwen Edit, Nano Banana, Flux Kontext)
* Video prompts
* Audio and music prompts
* System prompts
* Iterative prompt refinement and A/B testing
* Managing large prompt libraries without losing structure
* Collecting prompts directly from websites and AI platforms
* Optional local LLM-assisted drafting and refinement via Ollama

---

## 🌐 Firefox Browser Extension

Prompt Hub includes an official Firefox browser extension:

[PromptHub Connector for Firefox](https://addons.mozilla.org/en-GB/firefox/addon/prompthub-connector/?utm_source=chatgpt.com)

The extension allows you to:

* Save highlighted prompts directly from webpages
* Capture prompts from ChatGPT, Gemini, Claude, forums, blogs, and websites
* Review prompts before importing
* Classify prompts using:

  * Categories
  * Tools
  * Prompt Types
  * Tags
  * Groups / Projects
  * Notes
* Store webpage metadata and source URLs
* Send prompts directly to your local Prompt Hub instance

Designed for fast prompt collection during real-world AI workflows.

---

## 📱 iOS Companion Version

An iOS version of Prompt Hub is currently preparing for TestFlight release.

Recent desktop updates added improved compatibility for cross-platform prompt syncing and migration between desktop and iOS versions.

### Cross-Platform JSON Sync

Prompt Hub now supports:

* Shared PromptHub JSON export/import format
* Prompt library migration between desktop and iOS
* Structured category/tool preservation
* Tag preservation
* Metadata preservation
* Duplicate detection during import
* Local-first workflow support

The long-term goal is a seamless local-first prompt ecosystem across desktop and mobile devices.

---

## 🚀 Key Features

### Core Prompt Management

* Create, edit, delete, duplicate prompts
* Full-text search across:

  * Title
  * Content
  * Notes
  * Categories
  * Tools
  * Tags
* Free-form Categories and Tools
* Notes field for workflow context and usage tips

### Prompt Variants

* Linked variants for experimentation
* Variants hidden during normal browsing
* Family view for variant trees
* Parent-child relationships for prompt evolution

### Groups

* Optional group assignment per prompt
* Bulk group assignment
* Group-based filtering
* Organise projects and workflows cleanly

### Tags & Saved Views

* Free-form tagging system
* Saved sidebar views using query strings
* Fast filtering without complex setup

### Browser Capture Workflow

* Web-based prompt capture
* Prompt review before saving
* Metadata-aware imports
* Local extension communication

---

## 👁️ Vision-Assisted Drafting (Optional)

* Image-to-prompt drafting using local vision models
* Auto-thumbnail assignment
* Custom drafting instructions per image
* Fully local via Ollama-supported models

Useful for:

* Reverse-engineering image prompts
* Visual concept analysis
* Prompt drafting from screenshots or references

---

## ✏️ Using a Prompt

The Use screen supports:

* Editing before execution
* Saving refined versions
* Placeholder replacement
* Local LLM refinement
* Creating variants directly from the Use screen

---

## 🧩 Descriptor Packs & Modular Prompt Building

Prompt Hub supports reusable descriptor systems for:

* Characters
* Creatures
* Environments
* Props
* Vehicles
* Scenes

Features include:

* Descriptor packs
* Template-driven rendering
* Reusable prompt fragments
* Randomisation workflows
* AI-assisted remixing via Ollama

Designed for scalable prompt engineering workflows.

---

## 🔄 Import / Export Features

Prompt Hub supports:

* Full JSON prompt library export
* JSON-based library merging
* Database import/export
* ZIP backup and restore support
* Thumbnail/media preservation
* Cross-version compatibility improvements

Suitable for:

* Long-term prompt archiving
* Multi-device workflows
* Prompt sharing
* Offline AI research libraries
* Creative production pipelines

---

## ♾ Infinite Scroll

* IntersectionObserver-based loading
* Automatic pagination handling
* Organise mode preserved across loads
* Smooth browsing of large libraries

---

## 🎨 Theming

* CSS variable-based theme system
* No JS frameworks
* Theme persistence via localStorage
* Easily extendable

---

## 🛠 Requirements

* Python 3.10+
* Flask
* Pillow
* requests
* Ollama (optional for AI-assisted features)

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🗂 Project Structure

```text
.
├── app.py
├── prompts.db
├── requirements.txt
├── templates/
├── static/
├── temp/
└── env/
```

---

## 🔐 Security & Privacy

Prompt Hub is intended for local-first use.

* No telemetry
* No tracking
* No forced cloud integration
* No external API calls unless explicitly configured
* All prompt data stored locally

If exposing Prompt Hub externally, proper security configuration is your responsibility.

---

## 🧠 Optional Ollama Integration

Prompt Hub can integrate with locally hosted Ollama models for:

* Prompt refinement
* Prompt remixing
* Vision-assisted drafting
* Descriptor randomisation
* AI-assisted editing

Everything remains fully local.

Supported models depend on your Ollama installation.

### AI Actions and Knowledge Library

Open **AI** in the top navigation to manage reusable System Instructions, Prompt Templates,
AI Actions, and local Knowledge Collections. The upgraded **Use Prompt** screen runs the
selected action against the editable Final Prompt and always shows a review result before
anything is applied. Applying a result changes only the on-screen Final Prompt; saving a
variant or overwriting the stored prompt remains a separate, explicit action.

Knowledge documents are extracted, split into overlapping chunks, embedded with a
collection-specific Ollama embedding model, and stored in the existing SQLite database.
At execution time PromptHub embeds the task/current prompt, ranks chunks by cosine
similarity, and supplies only the best passages as clearly delimited reference context.
The Ollama host can be changed with `OLLAMA_HOST`; each collection has an editable
embedding model. See [AI Actions and Knowledge Library](docs/AI_ACTIONS_KNOWLEDGE.md)
for setup, usage, schema, testing, and limitations.

---

## Contributing

Contributions, improvements, and feature suggestions are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Open a Pull Request

Clear, focused improvements are preferred over large architectural rewrites.

---

## License

This project is licensed under the MIT License.

See the LICENSE file for details.
