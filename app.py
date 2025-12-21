import sqlite3
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"

PROMPT_TYPES = ["Generation", "Edit", "Instruction"]


# -------------------------
# DB helpers
# -------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()

    # Core lookup tables
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        """
    )

    # Main prompts table
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

    # Seed defaults once
    cat_count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    if cat_count == 0:
        for c in ["Image", "Video", "Music", "Other"]:
            conn.execute("INSERT INTO categories (name) VALUES (?)", (c,))

    tool_count = conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
    if tool_count == 0:
        for t in [
            "Generic",
            "Flux",
            "Flux Kontext",
            "Qwen Edit",
            "Nano Banana",
            "Z-Image",
            "Suno",
            "RVC",
            "Other",
        ]:
            conn.execute("INSERT INTO tools (name) VALUES (?)", (t,))

    conn.commit()
    conn.close()


def get_categories() -> list[str]:
    conn = get_db()
    rows = conn.execute("SELECT name FROM categories ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return [r["name"] for r in rows]


def get_tools() -> list[str]:
    conn = get_db()
    rows = conn.execute("SELECT name FROM tools ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    return [r["name"] for r in rows]


# -------------------------
# Prompt utilities
# -------------------------
def find_placeholders(text: str) -> list[str]:
    """Return a sorted list of [[placeholder]] names in the text."""
    return sorted(set(re.findall(r"\[\[(.+?)\]\]", text)))


def parse_prompt_txt(raw_text: str) -> dict:
    """
    Optional header lines at the top of the file (until '---' or first non-header):
      # title: ...
      # category: ...
      # tool: ...
      # type: Generation|Edit|Instruction
      # notes: ...
    """
    meta: dict[str, str] = {}
    lines = raw_text.splitlines()

    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped == "---":
            content_start = i + 1
            break

        if stripped.startswith("#") and ":" in stripped:
            # e.g. "# tool: Flux Kontext"
            try:
                key, val = stripped[1:].split(":", 1)
                meta[key.strip().lower()] = val.strip()
                content_start = i + 1
            except ValueError:
                # Ignore malformed header lines
                pass
        else:
            # First non-header line: stop parsing meta
            break

    content = "\n".join(lines[content_start:]).strip()

    return {
        "title": meta.get("title", ""),
        "category": meta.get("category", ""),
        "tool": meta.get("tool", ""),
        "prompt_type": meta.get("type", ""),
        "notes": meta.get("notes", ""),
        "content": content,
    }


# -------------------------
# Manage categories/tools
# -------------------------
@app.route("/manage", methods=["GET"])
def manage():
    categories = get_categories()
    tools = get_tools()
    return render_template("manage.html", categories=categories, tools=tools)


@app.route("/manage/category/add", methods=["POST"])
def add_category():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Category name cannot be empty.")
        return redirect(url_for("manage"))

    conn = get_db()
    try:
        conn.execute("INSERT INTO categories (name) VALUES (?)", (name,))
        conn.commit()
        flash(f"Added category: {name}")
    except sqlite3.IntegrityError:
        flash("That category already exists.")
    finally:
        conn.close()

    return redirect(url_for("manage"))


@app.route("/manage/category/delete", methods=["POST"])
def delete_category():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("manage"))

    conn = get_db()
    in_use = conn.execute("SELECT COUNT(*) FROM prompts WHERE category = ?", (name,)).fetchone()[0]
    if in_use > 0:
        conn.close()
        flash(f"Cannot delete '{name}' because it is used by {in_use} prompt(s).")
        return redirect(url_for("manage"))

    conn.execute("DELETE FROM categories WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    flash(f"Deleted category: {name}")
    return redirect(url_for("manage"))


@app.route("/manage/tool/add", methods=["POST"])
def add_tool():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Tool name cannot be empty.")
        return redirect(url_for("manage"))

    conn = get_db()
    try:
        conn.execute("INSERT INTO tools (name) VALUES (?)", (name,))
        conn.commit()
        flash(f"Added tool: {name}")
    except sqlite3.IntegrityError:
        flash("That tool already exists.")
    finally:
        conn.close()

    return redirect(url_for("manage"))


@app.route("/manage/tool/delete", methods=["POST"])
def delete_tool():
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("manage"))

    conn = get_db()
    in_use = conn.execute("SELECT COUNT(*) FROM prompts WHERE tool = ?", (name,)).fetchone()[0]
    if in_use > 0:
        conn.close()
        flash(f"Cannot delete '{name}' because it is used by {in_use} prompt(s).")
        return redirect(url_for("manage"))

    conn.execute("DELETE FROM tools WHERE name = ?", (name,))
    conn.commit()
    conn.close()
    flash(f"Deleted tool: {name}")
    return redirect(url_for("manage"))


# -------------------------
# Main pages / prompts CRUD
# -------------------------
@app.route("/", methods=["GET"])
def index():
    category = request.args.get("category", "all")
    tool = request.args.get("tool", "all")

    conn = get_db()
    query = "SELECT * FROM prompts WHERE 1=1"
    params: list[str] = []

    if category != "all":
        query += " AND category = ?"
        params.append(category)

    if tool != "all":
        query += " AND tool = ?"
        params.append(tool)

    query += " ORDER BY updated_at DESC"

    prompts = conn.execute(query, params).fetchall()
    conn.close()

    return render_template(
        "index.html",
        prompts=prompts,
        categories=get_categories(),
        tools=get_tools(),
        active_category=category,
        active_tool=tool,
    )


@app.route("/prompt/new", methods=["GET", "POST"])
def new_prompt():
    categories = get_categories()
    tools = get_tools()

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

        # Clamp to known lists (in case of stale form values)
        if category not in categories:
            category = "Other"
        if tool not in tools:
            tool = "Other"
        if prompt_type not in PROMPT_TYPES:
            prompt_type = "Instruction"

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
        prompt_types=PROMPT_TYPES,
    )


@app.route("/prompt/<int:prompt_id>/edit", methods=["GET", "POST"])
def edit_prompt(prompt_id: int):
    categories = get_categories()
    tools = get_tools()

    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    conn.close()

    if not prompt:
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

        if category not in categories:
            category = "Other"
        if tool not in tools:
            tool = "Other"
        if prompt_type not in PROMPT_TYPES:
            prompt_type = "Instruction"

        now = datetime.utcnow().isoformat()
        conn = get_db()
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

    return render_template(
        "edit_prompt.html",
        prompt=prompt,
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
    )


@app.route("/prompt/<int:prompt_id>/delete", methods=["POST"])
def delete_prompt(prompt_id: int):
    conn = get_db()
    conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()
    flash("Prompt deleted.")
    return redirect(url_for("index"))


@app.route("/prompt/<int:prompt_id>/render", methods=["GET", "POST"])
def render_prompt(prompt_id: int):
    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
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

        override = request.form.get("final_override", "").strip()
        if override:
            final_text = override

    return render_template(
        "render_prompt.html",
        prompt=prompt,
        placeholders=placeholders,
        final_text=final_text,
    )


# -------------------------
# TXT import
# -------------------------
@app.route("/prompt/import", methods=["GET", "POST"])
def import_prompt():
    categories = get_categories()
    tools = get_tools()

    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or all((f is None or f.filename == "") for f in files):
            flash("No files selected.")
            return redirect(url_for("import_prompt"))

        imported = 0
        skipped = 0

        conn = get_db()
        now = datetime.utcnow().isoformat()

        for f in files:
            if not f or not f.filename:
                continue

            filename = secure_filename(f.filename)
            if not filename.lower().endswith(".txt"):
                skipped += 1
                continue

            raw = f.read().decode("utf-8", errors="ignore")
            parsed = parse_prompt_txt(raw)

            title_default = filename.rsplit(".", 1)[0]

            title = parsed["title"].strip() or title_default
            category = parsed["category"].strip() or "Image"
            tool = parsed["tool"].strip() or "Generic"
            prompt_type = parsed["prompt_type"].strip() or "Instruction"
            notes = parsed["notes"].strip() or ""
            content = parsed["content"].strip()

            if not content:
                skipped += 1
                continue

            if category not in categories:
                category = "Other"
            if tool not in tools:
                tool = "Other"
            if prompt_type not in PROMPT_TYPES:
                prompt_type = "Instruction"

            conn.execute(
                """
                INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (title, category, tool, prompt_type, content, notes, now, now),
            )
            imported += 1

        conn.commit()
        conn.close()

        flash(f"Imported {imported} prompt(s). Skipped {skipped} file(s).")
        return redirect(url_for("index"))

    return render_template(
        "import_prompt.html",
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
    )


@app.route("/prompt/import/preview", methods=["POST"])
def import_prompt_preview():
    categories = get_categories()
    tools = get_tools()

    f = request.files.get("file")
    if not f or not f.filename:
        flash("No file selected.")
        return redirect(url_for("import_prompt"))

    filename = secure_filename(f.filename)
    if not filename.lower().endswith(".txt"):
        flash("Only .txt files are supported.")
        return redirect(url_for("import_prompt"))

    raw = f.read().decode("utf-8", errors="ignore")
    parsed = parse_prompt_txt(raw)

    title_default = filename.rsplit(".", 1)[0]
    prompt_like = {
        "title": parsed["title"].strip() or title_default,
        "category": parsed["category"].strip() or "Image",
        "tool": parsed["tool"].strip() or "Generic",
        "prompt_type": parsed["prompt_type"].strip() or "Instruction",
        "content": parsed["content"].strip(),
        "notes": parsed["notes"].strip(),
    }

    if prompt_like["category"] not in categories:
        prompt_like["category"] = "Other"
    if prompt_like["tool"] not in tools:
        prompt_like["tool"] = "Other"
    if prompt_like["prompt_type"] not in PROMPT_TYPES:
        prompt_like["prompt_type"] = "Instruction"

    return render_template(
        "edit_prompt.html",
        prompt=prompt_like,
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
        imported_preview=True,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
