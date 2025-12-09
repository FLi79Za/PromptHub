import sqlite3
import re
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            tool TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            content TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def find_placeholders(text: str):
    """Return a sorted list of [[placeholder]] names in the text."""
    return sorted(set(re.findall(r"\[\[(.+?)\]\]", text)))


@app.route("/")
def index():
    category = request.args.get("category", "all")
    tool = request.args.get("tool", "all")

    conn = get_db()

    query = "SELECT * FROM prompts WHERE 1=1"
    params = []

    if category != "all":
        query += " AND category = ?"
        params.append(category)

    if tool != "all":
        query += " AND tool = ?"
        params.append(tool)

    query += " ORDER BY updated_at DESC"

    prompts = conn.execute(query, params).fetchall()
    conn.close()

    categories = ["Image", "Video", "Music", "Other"]
    tools = ["Generic", "Flux", "Flux Kontext", "Qwen Edit", "Nano Banana", "Z-Image", "Suno", "RVC", "Other"]

    return render_template(
        "index.html",
        prompts=prompts,
        categories=categories,
        tools=tools,
        active_category=category,
        active_tool=tool,
    )


@app.route("/prompt/new", methods=["GET", "POST"])
def new_prompt():
    prompt_types = ["Generation", "Edit", "Instruction"]
    categories = ["Image", "Video", "Music", "Other"]
    tools = ["Generic", "Flux", "Flux Kontext", "Qwen Edit", "Nano Banana", "Z-Image", "Suno", "RVC", "Other"]

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content = request.form.get("content", "").strip()
        notes = request.form.get("notes", "").strip()

        if not title or not content:
            flash("Title and content are required.")
            return redirect(url_for("new_prompt"))

        now = datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute(
            """
            INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, category, tool, prompt_type, content, notes, now, now),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("index"))

    return render_template(
        "edit_prompt.html",
        prompt=None,
        categories=categories,
        tools=tools,
        prompt_types=prompt_types,
    )


@app.route("/prompt/<int:prompt_id>/edit", methods=["GET", "POST"])
def edit_prompt(prompt_id):
    prompt_types = ["Generation", "Edit", "Instruction"]
    categories = ["Image", "Video", "Music", "Other"]
    tools = ["Generic", "Flux", "Flux Kontext", "Qwen Edit", "Nano Banana", "Z-Image", "Suno", "RVC", "Other"]

    conn = get_db()
    prompt = conn.execute(
        "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
    ).fetchone()

    if not prompt:
        conn.close()
        flash("Prompt not found.")
        return redirect(url_for("index"))

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content = request.form.get("content", "").strip()
        notes = request.form.get("notes", "").strip()

        if not title or not content:
            flash("Title and content are required.")
            return redirect(url_for("edit_prompt", prompt_id=prompt_id))

        now = datetime.utcnow().isoformat()
        conn.execute(
            """
            UPDATE prompts
            SET title = ?, category = ?, tool = ?, prompt_type = ?, content = ?, notes = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, category, tool, prompt_type, content, notes, now, prompt_id),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("index"))

    conn.close()

    return render_template(
        "edit_prompt.html",
        prompt=prompt,
        categories=categories,
        tools=tools,
        prompt_types=prompt_types,
    )


@app.route("/prompt/<int:prompt_id>/delete", methods=["POST"])
def delete_prompt(prompt_id):
    conn = get_db()
    conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()
    flash("Prompt deleted.")
    return redirect(url_for("index"))


@app.route("/prompt/<int:prompt_id>/render", methods=["GET", "POST"])
def render_prompt(prompt_id):
    conn = get_db()
    prompt = conn.execute(
        "SELECT * FROM prompts WHERE id = ?", (prompt_id,)
    ).fetchone()
    conn.close()

    if not prompt:
        flash("Prompt not found.")
        return redirect(url_for("index"))

    content = prompt["content"]
    placeholders = find_placeholders(content)

    final_text = None

    if request.method == "POST":
        final_text = content
        for name in placeholders:
            value = request.form.get(name, "").strip()
            final_text = final_text.replace(f"[[{name}]]", value)

        # Also allow direct editing: if user overrides in the big textarea
        override = request.form.get("final_override", "").strip()
        if override:
            final_text = override

    return render_template(
        "render_prompt.html",
        prompt=prompt,
        placeholders=placeholders,
        final_text=final_text,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
