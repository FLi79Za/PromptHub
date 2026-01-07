# Prompt Hub (Prompt Library)

A local, self-hosted web app for storing, organising, searching, refining, and reusing AI prompts.

Prompt Hub is designed for **real-world AI workflows**, not toy examples. It supports full-sentence prompts, instruction-based editing, iterative refinement, and visual context — all running **entirely locally**.

**New:** Now features **Vision-Assisted Drafting** (reverse-engineer images to prompts), **Prompt Variants**, and **Smart Tagging**.

No cloud services required.

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone [https://github.com/](https://github.com/)<your-username>/<repo-name>.git
cd <repo-name>

# 2. Create virtual env
python -m venv env
# Windows:
env\Scripts\activate
# Mac/Linux:
# source env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
Open in your browser: 👉 http://127.0.0.1:5000

🧠 What this is good for
Image generation prompts (Flux, Stable Diffusion, Midjourney, etc.)

Instruction-based image editing (Qwen Edit, Nano Banana, Flux Kontext)

Reverse-engineering images into prompts using Vision AI

Prompt variants (A/B testing different styles)

Iterative refinement using a local LLM (Ollama)

🚀 Key Features
Core Prompt Management
Create, edit, delete, and duplicate prompts.

Prompt Variants: Create child "variants" of a prompt to iterate on ideas without losing the original.

Smart Tagging: Add free-form tags to prompts for flexible grouping.

Saved Views: Save your favorite filter combinations (e.g., "Flux Characters") to the sidebar for one-click access.

Dynamic Organization: Categories and Tools are created automatically when you type them—no manual setup needed.

👁️ Vision-Assisted Drafting (New!)
Image-to-Prompt: Drag an image into the "New Prompt" screen to auto-generate a prompt draft.

Two-Stage Pipeline: Uses a Vision model (e.g., LLaVA/Qwen-VL) to see and a Text model (e.g., Gemma) to write.

Auto-Thumbnail: The analyzed image is automatically set as the prompt's thumbnail.

Custom Instructions: Guide the AI on what to look for and how to write the prompt.

Visual Context
Thumbnails: Upload reference images for any prompt.

Drag-and-Drop: Drag an image directly onto a prompt card in the library to instantly update its thumbnail.

Optimization: Images are automatically resized and compressed to keep the app fast.

Search & Scale
Full-text search across Title, Content, Notes, Category, Tool, and Tags.

Pagination: Handles large libraries efficiently (12 items per page).

Bulk Actions: Bulk update Category, Tool, Type, and Tags for multiple prompts at once.

Local LLM Integration (Ollama)
Refine Prompts: Reword, shorten, or expand prompts locally.

Drafting: Generate prompt drafts from scratch or images.

Zero Setup: Automatically detects your local Ollama models.

🛠 Requirements
Python 3.10+ (3.11 recommended)

Git (optional, but recommended)

Ollama (Optional, required for AI refinement/vision features)

Python Dependencies
Flask (Web framework)

Pillow (Image processing)

requests (API calls)

📘 Usage Overview
Creating Prompts
Click New Prompt.

Vision Draft (Optional): Drag an image into the "Generate Draft" box. Select your local Vision and Text models (via Ollama) and click Generate.

Fill in details: Title, Category, Tool, Tags.

Use [[placeholders]] for variable sections.

Example: Change only the [[target_area]] so it becomes [[new_look]].

Using a Prompt
Click Use on any card.

Fill in the placeholders in the form.

Click Copy to Clipboard.

Variants & Versioning
Make Variant: On the Edit screen, click "Make Variant" to create a linked copy.

Tracking: Variants show a badge on the card (e.g., 2 ⑂) indicating how many versions exist.

Importing Prompts
Import individual .txt files or batch import multiple files.

Supports metadata headers in files:

YAML

# title: Flux Kontext - Render Block
# category: Image
# tags: realistic, 8k, portrait
---
Render a full resolution image...
Backup & Restore
Go to Manage.

Download Backup: Get a ZIP file containing your database (prompts.db) and all thumbnail images.

Restore: Upload a backup ZIP to restore your library perfectly.

🧠 Ollama Integration (Optional)
To enable AI features, install Ollama and pull models:

Bash

# Text model for drafting/refining
ollama pull gemma3:4b

# Vision model for image analysis
ollama pull llava
# OR
ollama pull qwen2.5-vl
The app will automatically detect these models in the dropdowns.

🗂 Project Structure
Plaintext

.
├── app.py                  # Main application logic
├── prompts.db              # Local SQLite database (not committed)
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates (Jinja2)
│   ├── index.html          # Library view with Filters/Sidebar
│   ├── edit_prompt.html    # Create/Edit + Vision Drafting
│   ├── render_prompt.html  # Usage view
│   └── ...
├── static/
│   ├── styles.css          # CSS Styling
│   └── uploads/thumbs/     # Stored thumbnails
└── env/                    # Virtual environment
🔐 Security Notes
This app is intended for local use. If you expose it beyond your local machine:

Add authentication.

Use HTTPS or a VPN.

Do not expose publicly without protection.