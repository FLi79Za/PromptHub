import sqlite3
import re
import uuid
import shutil
import requests
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"

PROMPT_TYPES = ["Generation", "Edit", "Instruction"]

# Thumbnails are stored on disk (recommended), DB stores relative path under /static
UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# GIF allowed (handy as a “video thumbnail” surrogate if you ever want it)
ALLOWED_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Limit upload size (adjust if you want)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15MB


# -------------------------
# DB helpers
# -------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c["name"] == column for c in cols)


def init_db() -> None:
    conn = get_db()

    # Lookup tables
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

    # Prompts table
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

    # Simple migration: add thumbnail column if missing
    if not column_exists(conn, "prompts", "thumbnail"):
        conn.execute("ALTER TABLE prompts ADD COLUMN thumbnail TEXT")

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
# Thumbnail helpers
# -------------------------
def is_allowed_thumb(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_THUMB_EXTS


def save_thumbnail(file_storage) -> str:
    """
    Saves uploaded thumbnail into static/uploads/thumbs/
    Returns DB-relative path from /static, e.g. 'uploads/thumbs/<uuid>.png'
    """
    original = secure_filename(file_storage.filename)
    ext = Path(original).suffix.lower()

    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / new_name
    file_storage.save(dest)

    return f"uploads/thumbs/{new_name}"


def copy_thumbnail(existing_rel_path: str) -> str | None:
    """
    Copies an existing thumbnail file to a new filename.
    Returns new relative path or None if missing.
    """
    if not existing_rel_path:
        return None

    src = BASE_DIR / "static" / existing_rel_path
    if not src.exists():
        return None

    ext = src.suffix.lower()
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / new_name
    shutil.copyfile(src, dest)
    return f"uploads/thumbs/{new_name}"


def delete_thumbnail_if_unused(rel_path: str) -> None:
    """
    Deletes the thumbnail file only if no other prompts reference it.
    """
    if not rel_path:
        return

    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM prompts WHERE thumbnail = ?",
        (rel_path,),
    ).fetchone()[0]
    conn.close()

    # If at least 2 prompts reference it, don't delete
    if count and count > 1:
        return

    abs_path = BASE_DIR / "static" / rel_path
    try:
        if abs_path.exists():
            abs_path.unlink()
    except Exception:
        pass


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
            try:
                key, val = stripped[1:].split(":", 1)
                meta[key.strip().lower()] = val.strip()
                content_start = i + 1
            except ValueError:
                pass
        else:
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
    return render_template("manage.html", categories=get_categories(), tools=get_tools())


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
    q = (request.args.get("q") or "").strip()

    categories = get_categories()
    tools = get_tools()

    conn = get_db()
    query = "SELECT * FROM prompts WHERE 1=1"
    params: list[str] = []

    if category != "all":
        query += " AND category = ?"
        params.append(category)

    if tool != "all":
        query += " AND tool = ?"
        params.append(tool)

    if q:
        query += """
            AND (
                LOWER(title) LIKE ?
                OR LOWER(content) LIKE ?
                OR LOWER(COALESCE(notes, '')) LIKE ?
                OR LOWER(tool) LIKE ?
                OR LOWER(category) LIKE ?
                OR LOWER(prompt_type) LIKE ?
            )
        """
        like = f"%{q.lower()}%"
        params.extend([like, like, like, like, like, like])

    query += " ORDER BY updated_at DESC"

    prompts = conn.execute(query, params).fetchall()
    conn.close()

    return render_template(
        "index.html",
        prompts=prompts,
        categories=categories,
        tools=tools,
        active_category=category,
        active_tool=tool,
        q=q,
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

        # Clamp to known lists
        if category not in categories:
            category = "Other"
        if tool not in tools:
            tool = "Other"
        if prompt_type not in PROMPT_TYPES:
            prompt_type = "Instruction"

        # Thumbnail upload (optional)
        thumbnail_path = None
        thumb_file = request.files.get("thumbnail_file")
        if thumb_file and thumb_file.filename:
            if not is_allowed_thumb(thumb_file.filename):
                flash("Thumbnail must be PNG, JPG, WebP, or GIF.")
                return redirect(url_for("new_prompt"))
            thumbnail_path = save_thumbnail(thumb_file)

        now = datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute(
            """
            INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, category, tool, prompt_type, content, notes, thumbnail_path, now, now),
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

        # Thumbnail logic
        new_thumbnail = prompt["thumbnail"]
        remove_thumb = request.form.get("remove_thumbnail") == "on"
        thumb_file = request.files.get("thumbnail_file")

        if remove_thumb and new_thumbnail:
            delete_thumbnail_if_unused(new_thumbnail)
            new_thumbnail = None

        if thumb_file and thumb_file.filename:
            if not is_allowed_thumb(thumb_file.filename):
                flash("Thumbnail must be PNG, JPG, WebP, or GIF.")
                return redirect(url_for("edit_prompt", prompt_id=prompt_id))

            # Replacing existing thumbnail -> delete if unused elsewhere
            if new_thumbnail:
                delete_thumbnail_if_unused(new_thumbnail)

            new_thumbnail = save_thumbnail(thumb_file)

        now = datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute(
            """
            UPDATE prompts
            SET title = ?, category = ?, tool = ?, prompt_type = ?, content = ?, notes = ?, thumbnail = ?, updated_at = ?
            WHERE id = ?
            """,
            (title, category, tool, prompt_type, content, notes, new_thumbnail, now, prompt_id),
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
    row = conn.execute("SELECT thumbnail FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    thumb = row["thumbnail"] if row else None

    conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.commit()
    conn.close()

    if thumb:
        delete_thumbnail_if_unused(thumb)

    flash("Prompt deleted.")
    return redirect(url_for("index"))


@app.route("/prompt/<int:prompt_id>/duplicate", methods=["POST"])
def duplicate_prompt(prompt_id: int):
    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()

    if not prompt:
        conn.close()
        flash("Prompt not found.")
        return redirect(url_for("index"))

    now = datetime.utcnow().isoformat()
    new_title = f"Copy of {prompt['title']}"

    # Copy thumbnail file (so delete/edit won’t affect the original)
    new_thumb = copy_thumbnail(prompt["thumbnail"]) if prompt["thumbnail"] else None

    conn.execute(
        """
        INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_title,
            prompt["category"],
            prompt["tool"],
            prompt["prompt_type"],
            prompt["content"],
            prompt["notes"],
            new_thumb,
            now,
            now,
        ),
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()

    flash("Prompt duplicated. You can edit the copy now.")
    return redirect(url_for("edit_prompt", prompt_id=new_id))


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
                INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (title, category, tool, prompt_type, content, notes, None, now, now),
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
        "thumbnail": None,
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



# ---------------------------
# Optional Ollama integration
# ---------------------------
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_list_models():
    """Return a list of locally available Ollama model names. Returns [] if Ollama isn't reachable."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        r.raise_for_status()
        data = r.json()
        models = []
        for m in data.get("models", []):
            name = m.get("name")
            if name:
                models.append(name)
        return models
    except Exception:
        return []

def ollama_generate(prompt: str, model: str | None = None, system: str | None = None) -> str:
    """Generate text via Ollama. Raises on failure."""
    payload = {
        "model": model or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def get_prompt(prompt_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,))
        return cur.fetchone()

def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ---------------------------
# Backup / Restore database
# ---------------------------
@app.route("/backup/download")
def backup_download():
    init_db()
    if not DB_PATH.exists():
        flash("Database file not found.")
        return redirect(url_for("manage"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompts_backup_{ts}.db"
    return send_file(DB_PATH, as_attachment=True, download_name=filename)

@app.route("/backup/restore", methods=["GET", "POST"])
def restore_backup():
    init_db()
    if request.method == "GET":
        return render_template("restore.html")
    f = request.files.get("db_file")
    if not f or f.filename.strip() == "":
        flash("No file selected.")
        return redirect(url_for("restore_backup"))
    # basic extension check
    filename = f.filename.lower()
    if not (filename.endswith(".db") or filename.endswith(".sqlite") or filename.endswith(".sqlite3")):
        flash("Please upload a .db / .sqlite file.")
        return redirect(url_for("restore_backup"))
    tmp_path = BASE_DIR / f"restore_{uuid.uuid4().hex}.db"
    f.save(tmp_path)
    # quick sanity: ensure it has prompts table
    try:
        with sqlite3.connect(tmp_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prompts'")
            if not cur.fetchone():
                raise ValueError("No prompts table")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        flash("That file doesn't look like a Prompt Hub database.")
        return redirect(url_for("restore_backup"))

    # backup current db first
    if DB_PATH.exists():
        bk = BASE_DIR / f"prompts_before_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2(DB_PATH, bk)
    shutil.copy2(tmp_path, DB_PATH)
    tmp_path.unlink(missing_ok=True)
    flash("Database restored successfully.")
    return redirect(url_for("manage"))

# ---------------------------
# Raw prompt API for preview/copy
# ---------------------------
@app.route("/prompt/<int:prompt_id>/raw", methods=["GET"])
def prompt_raw(prompt_id: int):
    p = get_prompt(prompt_id)
    if not p:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "id": p["id"],
        "title": p["title"],
        "content": p["content"] or "",
        "category": p["category"],
        "tool": p["tool"],
        "prompt_type": p["prompt_type"],
    })

# ---------------------------
# Bulk update (category/tool/type) from index
# ---------------------------
@app.route("/bulk_update", methods=["POST"])
def bulk_update():
    ids = request.form.getlist("prompt_ids")
    if not ids:
        flash("No prompts selected.")
        return redirect(url_for("index"))

    new_category = request.form.get("bulk_category") or ""
    new_tool = request.form.get("bulk_tool") or ""
    new_type = request.form.get("bulk_prompt_type") or ""

    # nothing to change
    if not any([new_category, new_tool, new_type]):
        flash("No bulk changes selected.")
        return redirect(url_for("index"))

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        for pid in ids:
            fields = []
            params = []
            if new_category:
                fields.append("category = ?")
                params.append(new_category)
            if new_tool:
                fields.append("tool = ?")
                params.append(new_tool)
            if new_type:
                fields.append("prompt_type = ?")
                params.append(new_type)
            fields.append("updated_at = ?")
            params.append(now_iso())
            params.append(int(pid))
            cur.execute(f"UPDATE prompts SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
    flash(f"Updated {len(ids)} prompt(s).")
    return redirect(url_for("index"))

# ---------------------------
# Ollama refine (saved prompt)
# ---------------------------
@app.route("/prompt/<int:prompt_id>/refine", methods=["GET", "POST"])
def refine_prompt(prompt_id: int):
    init_db()
    prompt = get_prompt(prompt_id)
    if not prompt:
        flash("Prompt not found.")
        return redirect(url_for("index"))

    models = ollama_list_models()
    ollama_available = bool(models)
    if request.method == "GET":
        return render_template("refine_prompt.html", prompt=prompt, models=models, ollama_available=ollama_available)

    if not ollama_available:
        flash("Ollama not available. Start Ollama and try again.")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    session["ollama_model"] = model
    instruction = (request.form.get("instruction") or "").strip()
    mode = request.form.get("mode") or "refine"  # refine | rewrite | expand

    base = prompt["content"] or ""
    sys = "You help refine AI prompts. Keep placeholders like [[name]] intact. Output only the prompt text."
    user_prompt = f"MODE: {mode}\nINSTRUCTION: {instruction}\n\nPROMPT:\n{base}"

    try:
        refined = ollama_generate(user_prompt, model=model, system=sys)
    except Exception as e:
        flash(f"Ollama error: {e}")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    save_as_new = request.form.get("save_as_new") == "on"
    if save_as_new:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"{prompt['title']} (refined)",
                    prompt["category"],
                    prompt["tool"],
                    prompt["prompt_type"],
                    refined,
                    prompt["notes"],
                    now_iso(),
                    now_iso(),
                ),
            )
            conn.commit()
        flash("Refined prompt saved as new.")
        return redirect(url_for("index"))

    # overwrite existing content
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE prompts SET content = ?, updated_at = ? WHERE id = ?", (refined, now_iso(), prompt_id))
        conn.commit()
    flash("Prompt refined.")
    return redirect(url_for("render_prompt", prompt_id=prompt_id))

# ---------------------------
# Ollama refine draft (new prompt stage)
# ---------------------------
@app.route("/refine_draft", methods=["POST"])
def refine_draft():
    init_db()
    models = ollama_list_models()
    ollama_available = bool(models)
    if not ollama_available:
        # render edit template again with message
        categories = get_categories()
        tools = get_tools()
        refine_error = "Ollama not available. Start Ollama and try again."
        prompt_like = {
            "id": None,
            "title": request.form.get("title", ""),
            "category": request.form.get("category", categories[0] if categories else "Other"),
            "tool": request.form.get("tool", tools[0] if tools else "Other"),
            "prompt_type": request.form.get("prompt_type", "Instruction"),
            "content": request.form.get("content", ""),
            "notes": request.form.get("notes", ""),
        }
        return render_template(
            "edit_prompt.html",
            prompt=prompt_like,
            categories=categories,
            tools=tools,
            prompt_types=sorted(list({ "Generation","Edit","Instruction"} | set([prompt_like["prompt_type"]]))),
            refine_error=refine_error,
            ollama_available=False,
        )

    model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    session["ollama_model"] = model
    instruction = (request.form.get("instruction") or "").strip()
    mode = request.form.get("mode") or "refine"
    base = request.form.get("content") or ""

    sys = "You help refine AI prompts. Keep placeholders like [[name]] intact. Output only the prompt text."
    user_prompt = f"MODE: {mode}\nINSTRUCTION: {instruction}\n\nPROMPT:\n{base}"

    refined = ""
    try:
        refined = ollama_generate(user_prompt, model=model, system=sys)
    except Exception as e:
        refined = ""
        refine_error = f"Ollama error: {e}"
    else:
        refine_error = None

    categories = get_categories()
    tools = get_tools()
    prompt_like = {
        "id": None,
        "title": request.form.get("title", ""),
        "category": request.form.get("category", categories[0] if categories else "Other"),
        "tool": request.form.get("tool", tools[0] if tools else "Other"),
        "prompt_type": request.form.get("prompt_type", "Instruction"),
        "content": base,
        "notes": request.form.get("notes", ""),
    }

    return render_template(
        "edit_prompt.html",
        prompt=prompt_like,
        categories=categories,
        tools=tools,
        prompt_types=sorted(list({ "Generation","Edit","Instruction"} | set([prompt_like["prompt_type"]]))),
        refined_draft=refined if refined else None,
        refine_error=refine_error,
        ollama_available=True,
    )


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
