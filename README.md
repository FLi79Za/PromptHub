# Prompt Library (Prompt Hub)

A local, self-hosted web app for storing, organising, searching, and reusing AI prompts.

Designed for real-world AI workflows, including:
- Image generation prompts (Flux, SD, etc.)
- Instruction-based image editing prompts (Qwen Edit, Nano Banana, Flux Kontext)
- Video, music, and other AI tool prompts
- Prompt variants and experimentation

Everything runs locally. No cloud services required.

---

## Features

- Create, edit, delete, and duplicate prompts
- Organise prompts by **Category** and **Tool**
- Add new categories and tools via the UI (no code changes)
- Instruction-style prompts supported (not just templates)
- Placeholder support using `[[placeholders]]`
- Render prompts with live substitution
- Copy generated prompts to clipboard
- Import prompts from `.txt` files (preview or bulk import)
- Full-text search across title, content, notes, tool, and category
- SQLite database (local, lightweight)

---

## Requirements

- **Python 3.10+** (3.11 recommended)
- Git (optional, but recommended)

## Check Python version:
```bash
python --version

## Setup (Windows 11)
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
If you don’t have requirements.txt yet, it should contain:

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

You should see output like:

nginx
Copy code
Running on http://127.0.0.1:5000
5) Open in your browser
Open:

cpp
Copy code
http://127.0.0.1:5000
Usage Overview
Creating prompts
Click New Prompt

Fill in title, category, tool, type, and content

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

Ideal for creating variations

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

Deletion is blocked if items are in use (to protect data)

Searching
Use the search bar on the main page

Searches across:

Title

Content

Notes

Tool

Category

Prompt type

Combine with category and tool filters

Project Structure
pgsql
Copy code
.
├── app.py
├── prompts.db          # Local database (not committed)
├── requirements.txt
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── edit_prompt.html
│   ├── render_prompt.html
│   ├── import_prompt.html
│   └── manage.html
├── static/
│   └── styles.css
└── env/                # Virtual environment (not committed)
Git Notes
Ensure your .gitignore includes:

bash
Copy code
env/
venv/
__pycache__/
*.pyc
prompts.db
This prevents committing your local database or virtual environment.

Security Notes
This app is intended for local use.

If you expose it beyond your local machine:

Add authentication

Use HTTPS or a VPN

Do not expose it publicly without protection

Roadmap Ideas (Optional)
Tags (free-form)

Export prompts back to .txt

Prompt version history

Mobile-friendly layout / PWA

Cloud or VPN-based access
