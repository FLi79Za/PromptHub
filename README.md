# Prompt Hub

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

The token, Flask session secret, API settings, and integration logs are stored outside the repository in `%LOCALAPPDATA%\PromptHub` by default. The server remains bound to `127.0.0.1` unless an allowed local-loopback value is configured.

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
