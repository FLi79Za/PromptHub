# Prompt Hub

A local, self-hosted web application for storing, organising, refining, and reusing AI prompts.

Prompt Hub is built for real-world AI workflows — not toy examples. It supports full-length prompts, instruction-based editing, iterative refinement, visual context, and structured organisation.

Everything runs entirely locally.

No cloud services required.

---

## Why Prompt Hub?

As prompt libraries grow, they become messy, duplicated, and hard to search.

Prompt Hub helps you:

- Organise complex prompts
- Maintain structured experimentation
- Iterate safely with variants
- Keep visual and contextual references
- Run optional local LLM refinement via Ollama

It is designed for serious AI users managing real prompt systems.

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
- Clean parent-child relationships

### Groups

- Optional single-group assignment per prompt
- Bulk group assignment
- Group-based filtering

### Tags & Saved Views

- Free-form tagging (many-to-many)
- Saved sidebar views using query strings
- Fast filtering without complex setup

---

## 👁️ Vision-Assisted Drafting (Optional)

- Image-to-prompt drafting using local vision models
- Auto-thumbnail assignment
- Custom drafting instructions per image
- Fully local via Ollama-supported models

---

## ✏️ Using a Prompt

The Use screen supports:

- Editing before execution
- Saving refined versions
- Placeholder replacement
- Local LLM refinement
- Creating new variants directly from the Use view

---

## ♾ Infinite Scroll

- IntersectionObserver-based loading
- Automatic pagination handling
- Organise mode preserved across loads
- Smooth browsing of large libraries

---

## 🎨 Theming

- CSS variable-based theme system
- No JS frameworks
- Theme persistence via localStorage
- Easily extendable

---

## 🛠 Requirements

- Python 3.10+
- Flask
- Pillow
- requests
- Ollama (optional for AI-assisted features)

Install dependencies with:

```bash
pip install -r requirements.txt
```

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

## 🔐 Security & Privacy

Prompt Hub is intended for local use.

- No telemetry
- No external API calls unless explicitly configured
- No background cloud dependencies
- All data stored locally

If exposing externally, proper security configuration is your responsibility.

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

This project is licensed under the MIT License — see the LICENSE file for details.