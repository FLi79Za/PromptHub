import sqlite3
import re
import uuid
import shutil
import io
import zipfile
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

# Optional local LLM integration (Ollama)
ENABLE_OLLAMA = True
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"

# Backups (DB + thumbnails)
BACKUP_DIR = BASE_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
THUMBS_DIR = BASE_DIR / "static" / "uploads" / "thumbs"


@app.context_processor
def inject_flags():
    return {"ENABLE_OLLAMA": ENABLE_OLLAMA, "DEFAULT_OLLAMA_MODEL": DEFAULT_OLLAMA_MODEL}

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



def ollama_list_models():
    """Return a sorted list of local Ollama model tags, or [] if unavailable."""
    if not ENABLE_OLLAMA:
        return []
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        models = sorted(set(models), key=str.lower)
        # Pin default model to top if present
        if DEFAULT_OLLAMA_MODEL in models:
            models.remove(DEFAULT_OLLAMA_MODEL)
            models.insert(0, DEFAULT_OLLAMA_MODEL)
        return models
    except Exception:
        return []


def ollama_generate(model: str, prompt: str, system: str = "") -> str:
    """Generate text from Ollama /api/generate. Raises on hard failure."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return (data.get("response") or "").strip()


def is_valid_prompthub_db(db_path: Path) -> bool:
    """Light validation to avoid restoring a random sqlite DB."""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        conn.close()
        required = {"prompts", "categories", "tools"}
        return required.issubset(tables)
    except Exception:
        return False


def split_into_paragraph_prompts(raw_text: str):
    """Split text into prompt chunks using blank lines as separators."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    # split on 1+ blank lines
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


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


# -------------------------
# Backup / Restore (DB + thumbnails)
# -------------------------
@app.route("/backup", methods=["GET"])
def backup_download():
    """Download a ZIP backup containing prompts.db and thumbnail files."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    zip_name = f"prompthub_backup_{ts}.zip"

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if DB_PATH.exists():
            z.write(DB_PATH, arcname="prompts.db")
        if THUMBS_DIR.exists():
            for p in THUMBS_DIR.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(Path("thumbs") / p.name))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name=zip_name, mimetype="application/zip")


@app.route("/restore", methods=["GET", "POST"])
def restore_backup():
    """Restore from a Prompt Hub backup ZIP."""
    if request.method == "GET":
        return render_template("restore.html")

    f = request.files.get("backup_file")
    if not f or not f.filename:
        flash("No backup file selected.")
        return redirect(url_for("restore_backup"))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    uploaded_zip = BACKUP_DIR / f"uploaded_restore_{ts}.zip"
    f.save(uploaded_zip)

    # Safety backup first
    safety_zip = BACKUP_DIR / f"safety_backup_before_restore_{ts}.zip"
    try:
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
            if DB_PATH.exists():
                z.write(DB_PATH, arcname="prompts.db")
            if THUMBS_DIR.exists():
                for p in THUMBS_DIR.rglob("*"):
                    if p.is_file():
                        z.write(p, arcname=str(Path("thumbs") / p.name))
        mem.seek(0)
        safety_zip.write_bytes(mem.read())
    except Exception as e:
        flash(f"Could not create safety backup: {e}")
        return redirect(url_for("restore_backup"))

    extract_dir = BACKUP_DIR / f"restore_extract_{ts}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(uploaded_zip, "r") as z:
            z.extractall(extract_dir)
    except Exception as e:
        flash(f"Invalid ZIP file: {e}")
        return redirect(url_for("restore_backup"))

    candidate_db = extract_dir / "prompts.db"
    if not candidate_db.exists():
        flash("Backup ZIP does not contain prompts.db")
        return redirect(url_for("restore_backup"))

    if not is_valid_prompthub_db(candidate_db):
        flash("The uploaded database does not look like a valid Prompt Hub database.")
        return redirect(url_for("restore_backup"))

    try:
        shutil.copyfile(candidate_db, DB_PATH)
    except Exception as e:
        flash(f"Failed to restore database: {e}")
        return redirect(url_for("restore_backup"))

    thumbs_src = extract_dir / "thumbs"
    try:
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        if thumbs_src.exists():
            for p in thumbs_src.rglob("*"):
                if p.is_file():
                    shutil.copyfile(p, THUMBS_DIR / p.name)
    except Exception as e:
        flash(f"Database restored, but thumbnails restore had issues: {e}")
        return redirect(url_for("index"))

    flash("Restore complete. Your database has been replaced. A safety backup ZIP was created in /backups.")
    return redirect(url_for("index"))

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


# -------------------------
# Optional: Ollama prompt refinement (existing prompts)
# -------------------------
@app.route("/prompt/<int:prompt_id>/refine", methods=["GET", "POST"])
def refine_prompt(prompt_id):
    if not ENABLE_OLLAMA:
        flash("Ollama integration is disabled.")
        return redirect(url_for("edit_prompt", prompt_id=prompt_id))

    prompt = get_prompt(prompt_id)
    if not prompt:
        flash("Prompt not found.")
        return redirect(url_for("index"))

    models = ollama_list_models()
    default_model = session.get("ollama_model", DEFAULT_OLLAMA_MODEL)
    refined = None
    error = None

    if request.method == "POST":
        model = (request.form.get("model") or default_model).strip()
        session["ollama_model"] = model
        instruction = (request.form.get("instruction") or "").strip()
        mode = (request.form.get("mode") or "refine").strip()

        system = "You are a prompt engineering assistant. Output ONLY the updated prompt text. No explanations."
        user_prompt = f"""TASK: {mode}
INSTRUCTION: {instruction}

SOURCE PROMPT:
{prompt['content']}

RULES:
- Keep the intent unless instruction asks otherwise.
- Preserve placeholders like [[like_this]] exactly.
- Output only the final prompt text.
""".strip()

        try:
            refined = ollama_generate(model=model, prompt=user_prompt, system=system)
            if not refined:
                error = "LLM returned an empty response."
        except Exception as e:
            error = f"Ollama error: {e}"

        default_model = session.get("ollama_model", DEFAULT_OLLAMA_MODEL)

    return render_template(
        "refine_prompt.html",
        prompt=prompt,
        models=models,
        default_model=default_model,
        refined=refined,
        error=error,
    )

@app.route("/prompt/import", methods=["GET", "POST"])

def import_prompt():
    categories = get_categories()
    tools = get_tools()

    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or all((f is None or f.filename == "") for f in files):
            flash("No files selected.")
            return redirect(url_for("import_prompt"))

        # Batch-applied defaults
        batch_category = (request.form.get("batch_category") or "").strip()
        batch_tool = (request.form.get("batch_tool") or "").strip()
        batch_prompt_type = (request.form.get("batch_prompt_type") or "").strip()
        batch_notes = (request.form.get("batch_notes") or "").strip()

        # Behaviour toggles
        single_prompt_per_file = (request.form.get("single_prompt_per_file") == "on")
        use_first_line_as_title = (request.form.get("use_first_line_as_title") != "off")  # default ON
        prefer_file_metadata = (request.form.get("prefer_file_metadata") == "on")  # optional

        imported = 0
        skipped = 0

        conn = get_db()
        now = datetime.utcnow().isoformat()

        for f in files:
            if not f or not f.filename:
                skipped += 1
                continue

            filename = secure_filename(f.filename)
            if not filename.lower().endswith(".txt"):
                skipped += 1
                continue

            raw = f.read().decode("utf-8", errors="ignore")
            title_default = filename.rsplit(".", 1)[0].strip() or "Imported Prompt"

            # If user wants to prefer metadata inside the file, parse it first
            parsed = parse_prompt_txt(raw) if prefer_file_metadata else {
                "title": "",
                "category": "",
                "tool": "",
                "prompt_type": "",
                "notes": "",
                "content": raw,
            }

            # Base metadata resolution
            category = (parsed.get("category","") or batch_category or "Other").strip()
            tool = (parsed.get("tool","") or batch_tool or "Other").strip()
            prompt_type = (parsed.get("prompt_type","") or batch_prompt_type or "Instruction").strip()
            notes = ((parsed.get("notes","") or "").strip())
            if batch_notes:
                notes = (notes + "\n\n" + batch_notes).strip() if notes else batch_notes

            # Validate against existing lists
            if category not in categories:
                category = "Other"
            if tool not in tools:
                tool = "Other"
            if prompt_type not in PROMPT_TYPES:
                prompt_type = "Instruction"

            content_blob = (parsed.get("content") or "").strip()
            if not content_blob:
                skipped += 1
                continue

            chunks = [content_blob] if single_prompt_per_file else split_into_paragraph_prompts(content_blob)
            if not chunks:
                skipped += 1
                continue

            for idx, chunk in enumerate(chunks, start=1):
                chunk_lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
                if not chunk_lines:
                    skipped += 1
                    continue

                if use_first_line_as_title:
                    title = chunk_lines[0].strip()
                    body = "\n".join(chunk_lines[1:]).strip()
                    if not body:
                        # fallback: keep the line as body too so we don't create empty prompts
                        body = title
                else:
                    title = f"{title_default} - {idx:02d}" if not single_prompt_per_file else title_default
                    body = chunk.strip()

                # If title is blank for some reason, fallback to filename numbering
                if not title:
                    title = f"{title_default} - {idx:02d}" if not single_prompt_per_file else title_default

                conn.execute(
                    """
                    INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (title, category, tool, prompt_type, body, notes, None, now, now),
                )
                imported += 1

        conn.commit()
        conn.close()

        flash(f"Imported {imported} prompt(s). Skipped {skipped} item(s).")
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


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
