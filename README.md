# Prompt Hub

A local, self-hosted web application for storing, organising, refining, and reusing AI prompts.

Prompt Hub is designed for **real-world AI workflows**, not toy examples. It supports full-length prompts, instruction-based editing, iterative refinement, visual context, and structured organisation — all running **entirely locally**.

No cloud services required.

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
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
http://127.0.0.1:5000

---

## 🧠 What Prompt Hub Is Good For

- Image generation prompts (Flux, Stable Diffusion, Midjourney, etc.)
- Instruction-based image editing (Qwen Edit, Nano Banana, Flux Kontext)
- Video, audio, and system prompts
- Iterative prompt refinement and A/B testing
- Managing large prompt libraries without losing structure
- Optional local LLM-assisted drafting and refinement (via Ollama)

---

## 🚀 Key Features

### Core Prompt Management
- Create, edit, delete, duplicate prompts
- Full-text search across title, content, notes, category, tool, and tags
- Free-form Categories and Tools
- Notes field for context and usage tips

### Prompt Variants
- Linked variants for experimentation
- Variants hidden in browsing, visible in search and family views

### Groups
- Optional single-group assignment per prompt
- Bulk group assignment
- Group-based filtering

### Tags & Saved Views
- Free-form tagging (many-to-many)
- Saved sidebar views using query strings

---

## 👁️ Vision-Assisted Drafting (Optional)

- Image-to-prompt drafting using local vision models
- Auto-thumbnail assignment
- Custom instructions per draft

---

## ✏️ Using a Prompt

- Use screen supports editing and saving
- Placeholder replacement
- Local LLM refinement
- Variant creation from Use view

---

## ♾ Infinite Scroll

- IntersectionObserver-based loading
- Pagination disabled automatically
- Organise mode preserved across loads

---

## 🎨 Theming

- CSS variable-based theme system
- No JS frameworks
- Theme persistence via localStorage

---

## 🛠 Requirements

- Python 3.10+
- Flask
- Pillow
- requests
- Ollama (optional)

---

## 🗂 Project Structure

```
.
├── app.py
├── prompts.db
├── requirements.txt
├── templates/
├── static/
└── env/
```

---

## 🔐 Security Notes

This app is intended for local use only.
