# Prompt Hub (Prompt Library)

A local, self-hosted web app for storing, organising, searching, refining, and reusing AI prompts.

Prompt Hub is designed for **real-world AI workflows**, not toy examples. It supports full-sentence prompts, instruction-based editing, iterative refinement, and visual context — all running **entirely locally**.

No cloud services required.

---

## ⚡ Quick Start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
python app.py
Open in your browser:
👉 http://127.0.0.1:5000

🧠 What this is good for
Image generation prompts (Flux, Stable Diffusion, etc.)

Instruction-based image editing prompts
(Qwen Edit, Nano Banana, Flux Kontext)

Video, music, and other AI tool prompts

Prompt variants and experimentation

Iterative refinement using a local LLM (Ollama)

🚀 Key Features
Core prompt management
Create, edit, delete, and duplicate prompts

Organise prompts by Category and Tool

Add new categories and tools via the UI (no code changes)

SQLite database (local, lightweight)

Prompt authoring & usage
Full-sentence prompts supported (not just templates)

Placeholder support using [[placeholders]]

Render prompts with live substitution

Edit final rendered output before copying

One-click copy to clipboard

Search & scale
Full-text search across:

Title

Content

Notes

Category

Tool

Prompt type

Combine search with category and tool filters

Designed to scale to large prompt libraries

Import / export
Import prompts from .txt files

Preview before saving

Bulk import supported

Optional metadata headers supported

Visual context
Optional thumbnail images per prompt

Useful for image and video prompts

Supports portrait, landscape, or square images

Thumbnails improve browsing and recognition

Local LLM integration (optional)
Ollama integration for prompt refinement

Runs entirely locally

Can be disabled without breaking the app

Supported workflows:

Refine an existing prompt

Refine a prompt at the “New Prompt” draft stage

Create variants (save as new prompts)

Convert between prompt styles (e.g. Flux ↔ instruction-based)

Shorten, expand, or rewrite prompts

The app remains fully usable even if Ollama is not installed or running.

🛠 Requirements
Python 3.10+ (3.11 recommended)

Git (optional, but recommended)

Windows 11 tested (should also work on macOS/Linux)

Check Python version:

bash
Copy code
python --version
🧩 Setup (Windows 11)
1) Clone the repository
bash
Copy code
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2) Create a virtual environment
bash
Copy code
python -m venv env
Activate it:

bash
Copy code
env\Scripts\activate
You should now see (env) in your terminal prompt.

3) Install dependencies
bash
Copy code
pip install -r requirements.txt
Minimum requirements.txt:

txt
Copy code
Flask>=3.0.0
4) Run the app
bash
Copy code
python app.py
On first run, the app will:

Create prompts.db

Create default categories and tools

You should see:

nginx
Copy code
Running on http://127.0.0.1:5000
📘 Usage Overview
Creating prompts
Click New Prompt

Fill in:

Title

Category

Tool

Prompt type

Content

Optionally add:

Notes

Thumbnail image

Use [[placeholders]] for variable sections

Example:

text
Copy code
Change only the [[target_area]] so it becomes [[new_look]].
Keep everything else the same.
Using a prompt
Click Use

Fill in placeholders

Optionally tweak the final text

Click Copy to clipboard

Duplicating prompts
Click Duplicate

A copy opens immediately for editing

Ideal for creating variations without overwriting originals

Importing prompts from .txt
Go to Import

Preview a single file or bulk import multiple files

Optional metadata header supported:

text
Copy code
# title: Flux Kontext – Render Block
# category: Image
# tool: Flux Kontext
# type: Instruction
# notes: Best results at 1024px
---
Render a full resolution image of block [[block_number]]...
Managing categories and tools
Go to Manage

Add or remove categories/tools via the UI

Deletion is blocked if items are in use (data safety)

Searching
Use the search bar on the main page.

Searches across:

Title

Content

Notes

Tool

Category

Prompt type

Combine with category and tool filters for large libraries.

🧠 Ollama Integration (Optional)
If you have Ollama installed and running:

bash
Copy code
ollama list
The app will:

Detect available models

Default to gemma3:4b

Remember the last model you selected

You can:

Refine prompts from Edit

Refine prompts from Use

Refine drafts before saving a new prompt

If Ollama is not installed or running:

The app still works normally

Refinement buttons are optional and safe

🗂 Project Structure
pgsql
Copy code
.
├── app.py
├── prompts.db              # Local database (not committed)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── edit_prompt.html
│   ├── render_prompt.html
│   ├── refine_prompt.html
│   ├── import_prompt.html
│   └── manage.html
├── static/
│   └── styles.css
└── env/                    # Virtual environment (not committed)
🧾 Git Notes
Ensure your .gitignore includes:

gitignore
Copy code
env/
venv/
__pycache__/
*.pyc
prompts.db
This prevents committing your local database or virtual environment.

🔐 Security Notes
This app is intended for local use.

If you expose it beyond your local machine:

Add authentication

Use HTTPS or a VPN

Do not expose it publicly without protection

🛣 Roadmap Ideas (Optional)
Tags (free-form)

Export prompts back to .txt

Prompt version history

Batch refinement

Mobile-friendly layout / PWA

Cloud or VPN-based access with sync

yaml
Copy code
