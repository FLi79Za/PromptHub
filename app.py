import sqlite3
import re
import uuid
import shutil
import requests
import zipfile
import io
import os
import math
import json
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageSequence

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"

PROMPT_TYPES = ["Generation", "Edit", "Instruction"]
UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
ITEMS_PER_PAGE = 12


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
    
    conn.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    conn.execute("CREATE TABLE IF NOT EXISTS tools (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            tool TEXT NOT NULL,
            prompt_type TEXT NOT NULL,
            content TEXT NOT NULL,
            notes TEXT,
            thumbnail TEXT,
            parent_id INTEGER,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            query_params TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    conn.execute("CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_tags (
            prompt_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (prompt_id, tag_id),
            FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE,
            FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
    """)

    if not column_exists(conn, "prompts", "thumbnail"):
        conn.execute("ALTER TABLE prompts ADD COLUMN thumbnail TEXT")
    if not column_exists(conn, "prompts", "parent_id"):
        conn.execute("ALTER TABLE prompts ADD COLUMN parent_id INTEGER")

    if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == 0:
        for c in ["Image", "Video", "Music", "Other"]: conn.execute("INSERT INTO categories (name) VALUES (?)", (c,))
    if conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0] == 0:
        for t in ["Generic", "Flux", "Flux Kontext", "Qwen Edit", "Nano Banana", "Z-Image", "Suno", "RVC", "Other"]:
            conn.execute("INSERT INTO tools (name) VALUES (?)", (t,))

    conn.commit()
    conn.close()

def get_categories() -> list[str]:
    with get_db() as conn: return [r["name"] for r in conn.execute("SELECT name FROM categories ORDER BY name COLLATE NOCASE")]

def get_tools() -> list[str]:
    with get_db() as conn: return [r["name"] for r in conn.execute("SELECT name FROM tools ORDER BY name COLLATE NOCASE")]

def get_all_tags() -> list[str]:
    with get_db() as conn: return [r["name"] for r in conn.execute("SELECT name FROM tags ORDER BY name ASC")]

def get_saved_views():
    with get_db() as conn: return conn.execute("SELECT * FROM saved_views ORDER BY name ASC").fetchall()

def get_prompt(prompt_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()

# -------------------------
# Tag Logic
# -------------------------
def save_tags(conn, prompt_id, tags_str):
    conn.execute("DELETE FROM prompt_tags WHERE prompt_id=?", (prompt_id,))
    if not tags_str: return

    raw_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    for tag_name in raw_tags:
        try: conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        except sqlite3.IntegrityError: pass
        
        tag_row = conn.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()
        if tag_row:
            try: conn.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (prompt_id, tag_row[0]))
            except sqlite3.IntegrityError: pass

def get_tags_for_prompt(conn, prompt_id):
    rows = conn.execute("""
        SELECT t.name FROM tags t
        JOIN prompt_tags pt ON t.id = pt.tag_id
        WHERE pt.prompt_id = ?
        ORDER BY t.name ASC
    """, (prompt_id,)).fetchall()
    return [r["name"] for r in rows]


# -------------------------
# Logic Helpers
# -------------------------
def ensure_category_tool_exist(category, tool):
    # Only opens a connection if it's called outside a transaction loop
    # For import_prompt, we use a local helper instead.
    try:
        conn = get_db()
        if category:
            try: conn.execute("INSERT INTO categories (name) VALUES (?)", (category,))
            except: pass
        if tool:
            try: conn.execute("INSERT INTO tools (name) VALUES (?)", (tool,))
            except: pass
        conn.commit()
        conn.close()
    except Exception:
        pass

def is_allowed_thumb(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_THUMB_EXTS

def save_thumbnail(file_storage) -> str:
    original_filename = secure_filename(file_storage.filename)
    ext = Path(original_filename).suffix.lower()
    file_id = uuid.uuid4().hex
    
    try:
        img = Image.open(file_storage)
        
        # --- GIF ANIMATION HANDLING ---
        if ext == '.gif' and getattr(img, "is_animated", False):
            new_name = f"{file_id}.webp"
            dest = UPLOAD_DIR / new_name
            
            frames = []
            durations = []
            
            try:
                # Limit frames to prevent huge files
                for i in range(60): 
                    img.seek(i)
                    frame_duration = img.info.get('duration', 100)
                    durations.append(frame_duration)
                    f = img.convert("RGBA")
                    f.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    frames.append(f)
            except EOFError:
                pass 

            if frames:
                frames[0].save(
                    dest,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=durations,
                    loop=0,
                    quality=85
                )
                return f"uploads/thumbs/{new_name}"
        
        # --- STATIC IMAGE ---
        new_name = f"{file_id}{ext}"
        dest = UPLOAD_DIR / new_name
        
        img.thumbnail((600, 600))
        
        if ext in ['.jpg', '.jpeg']:
            if img.mode != 'RGB': img = img.convert('RGB')
            img.save(dest, optimize=True, quality=85)
        elif ext == '.png':
            img.save(dest, optimize=True)
        elif ext == '.webp':
            img.save(dest, optimize=True, quality=85)
        else:
            img.save(dest)
            
        return f"uploads/thumbs/{new_name}"

    except Exception as e:
        print(f"Image processing failed: {e}")
        file_storage.seek(0)
        fallback_name = f"{uuid.uuid4().hex}{ext}"
        file_storage.save(UPLOAD_DIR / fallback_name)
        return f"uploads/thumbs/{fallback_name}"

def copy_thumbnail(existing_rel_path: str) -> str | None:
    if not existing_rel_path: return None
    src = BASE_DIR / "static" / existing_rel_path
    if not src.exists(): return None
    ext = src.suffix.lower()
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / new_name
    try:
        shutil.copyfile(src, dest)
        return f"uploads/thumbs/{new_name}"
    except:
        return None

def delete_thumbnail_if_unused(rel_path: str) -> None:
    if not rel_path: return
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM prompts WHERE thumbnail = ?", (rel_path,)).fetchone()[0]
    conn.close()
    if count <= 1:
        try: (BASE_DIR / "static" / rel_path).unlink()
        except: pass

def find_placeholders(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[\[(.+?)\]\]", text)))

def parse_prompt_txt(raw_text: str) -> dict:
    meta = {}
    lines = raw_text.splitlines()
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            content_start = i + 1; break
        if stripped.startswith("#") and ":" in stripped:
            try:
                key, val = stripped[1:].split(":", 1)
                meta[key.strip().lower()] = val.strip()
                content_start = i + 1
            except ValueError: pass
        else: break
    return {
        "title": meta.get("title", ""), "category": meta.get("category", ""),
        "tool": meta.get("tool", ""), "prompt_type": meta.get("type", ""),
        "notes": meta.get("notes", ""), "content": "\n".join(lines[content_start:]).strip(),
    }


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    category = request.args.get("category", "all")
    tool = request.args.get("tool", "all")
    tag_filter = request.args.get("tag", "all")
    q = (request.args.get("q") or "").strip()
    q_not = (request.args.get("q_not") or "").strip()
    sort_by = request.args.get("sort_by", "updated_at")
    sort_dir = request.args.get("sort_dir", "desc")
    try: page = max(1, int(request.args.get("page", 1)))
    except: page = 1

    ALLOWED_SORTS = {"updated_at", "created_at", "title", "category", "tool", "prompt_type"}
    if sort_by not in ALLOWED_SORTS: sort_by = "updated_at"
    if sort_dir not in ["asc", "desc"]: sort_dir = "desc"

    conn = get_db()
    
    # Base query logic
    base_query = " FROM prompts p"
    
    # We only JOIN here for the specific Tag dropdown filter (Single Tag Mode)
    if tag_filter != "all":
        base_query += " JOIN prompt_tags pt ON p.id = pt.prompt_id JOIN tags t ON pt.tag_id = t.id"
    
    base_query += " WHERE 1=1"
    params = []

    if category != "all": base_query += " AND p.category = ?"; params.append(category)
    if tool != "all": base_query += " AND p.tool = ?"; params.append(tool)
    if tag_filter != "all": base_query += " AND t.name = ?"; params.append(tag_filter)
    
    # --- UPDATED SEARCH LOGIC (Comma Separated + Tags) ---
    
    # 1. POSITIVE SEARCH (AND logic for multiple terms)
    if q:
        terms = [t.strip() for t in q.split(',') if t.strip()]
        for term in terms:
            # Check Title, Content, Notes OR Tags
            # Using EXISTS for tags prevents duplicates and logic errors
            sub_query = """
                (LOWER(p.title) LIKE ? OR 
                 LOWER(p.content) LIKE ? OR 
                 LOWER(COALESCE(p.notes, '')) LIKE ? OR
                 EXISTS (
                    SELECT 1 FROM prompt_tags pt_s 
                    JOIN tags t_s ON pt_s.tag_id = t_s.id 
                    WHERE pt_s.prompt_id = p.id AND LOWER(t_s.name) LIKE ?
                 )
                )
            """
            base_query += f" AND {sub_query}"
            like_term = f"%{term.lower()}%"
            params.extend([like_term, like_term, like_term, like_term])

    # 2. NEGATIVE SEARCH (AND NOT logic for multiple terms)
    if q_not:
        terms = [t.strip() for t in q_not.split(',') if t.strip()]
        for term in terms:
            # Exclude if Title, Content, Notes OR Tags match
            sub_query = """
                NOT (LOWER(p.title) LIKE ? OR 
                     LOWER(p.content) LIKE ? OR 
                     LOWER(COALESCE(p.notes, '')) LIKE ? OR
                     EXISTS (
                        SELECT 1 FROM prompt_tags pt_s 
                        JOIN tags t_s ON pt_s.tag_id = t_s.id 
                        WHERE pt_s.prompt_id = p.id AND LOWER(t_s.name) LIKE ?
                     )
                )
            """
            base_query += f" AND {sub_query}"
            like_term = f"%{term.lower()}%"
            params.extend([like_term, like_term, like_term, like_term])

    # --- END UPDATED LOGIC ---

    count_query = f"SELECT COUNT(DISTINCT p.id) {base_query}"
    total_items = conn.execute(count_query, params).fetchone()[0]
    total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
    
    offset = (page - 1) * ITEMS_PER_PAGE
    data_query = f"""
        SELECT DISTINCT p.*, 
        (SELECT COUNT(*) FROM prompts AS p2 WHERE p2.parent_id = p.id) as child_count 
        {base_query} 
        ORDER BY p.{sort_by} {sort_dir.upper()} 
        LIMIT ? OFFSET ?
    """
    rows = conn.execute(data_query, params + [ITEMS_PER_PAGE, offset]).fetchall()
    
    prompts = []
    for r in rows:
        p_dict = dict(r)
        p_dict["tags"] = get_tags_for_prompt(conn, p_dict["id"])
        prompts.append(p_dict)
    
    saved_views = get_saved_views()
    all_tags = get_all_tags()
    conn.close()

    return render_template(
        "index.html",
        prompts=prompts,
        categories=get_categories(),
        tools=get_tools(),
        active_category=category,
        active_tool=tool,
        active_tag=tag_filter,
        q=q,
        q_not=q_not,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        total_pages=total_pages,
        total_items=total_items,
        saved_views=saved_views,
        all_tags=all_tags
    )


@app.route("/api/prompt/<int:prompt_id>/upload_thumb", methods=["POST"])
def upload_thumb_api(prompt_id):
    f = request.files.get("file")
    if not f or not is_allowed_thumb(f.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    conn = get_db()
    row = conn.execute("SELECT thumbnail FROM prompts WHERE id=?", (prompt_id,)).fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Prompt not found"}), 404
        
    old_thumb = row["thumbnail"]
    new_path = save_thumbnail(f)
    
    if old_thumb: delete_thumbnail_if_unused(old_thumb)
    
    conn.execute("UPDATE prompts SET thumbnail=?, updated_at=? WHERE id=?", 
                 (new_path, datetime.utcnow().isoformat(), prompt_id))
    conn.commit()
    conn.close()
    
    return jsonify({"success": True, "thumbnail_url": url_for('static', filename=new_path)})


@app.route("/view/save", methods=["POST"])
def save_view():
    name = request.form.get("view_name", "").strip()
    query_string = request.form.get("query_string", "")
    if not name or not query_string:
        return redirect(url_for("index"))
    
    conn = get_db()
    conn.execute("INSERT INTO saved_views (name, query_params, created_at) VALUES (?, ?, ?)", 
                 (name, query_string, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    flash(f"View '{name}' saved.")
    return redirect(f"/?{query_string}")

@app.route("/view/delete/<int:view_id>", methods=["POST"])
def delete_view(view_id):
    conn = get_db()
    conn.execute("DELETE FROM saved_views WHERE id=?", (view_id,))
    conn.commit(); conn.close()
    return redirect(url_for("index"))


# -------------------------
# Prompt CRUD
# -------------------------
@app.route("/prompt/new", methods=["GET", "POST"])
def new_prompt():
    categories, tools = get_categories(), get_tools()
    ollama_models = ollama_list_models()

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        ensure_category_tool_exist(category, tool)

        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content, notes = request.form.get("content", "").strip(), request.form.get("notes", "").strip()
        tags_str = request.form.get("tags", "").strip()

        if not title or not content:
            flash("Title and content are required.")
            return redirect(url_for("new_prompt"))

        thumb_path = None
        f = request.files.get("thumbnail")
        hidden_thumb = request.form.get("hidden_thumbnail_path")

        if f and f.filename and is_allowed_thumb(f.filename):
            thumb_path = save_thumbnail(f)
        elif hidden_thumb:
            thumb_path = hidden_thumb

        now = datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute(
            """
            INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, category, tool, prompt_type, content, notes, thumb_path, now, now),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        save_tags(conn, new_id, tags_str)
        conn.commit(); conn.close()
        return redirect(url_for("index"))

    return render_template(
        "edit_prompt.html",
        prompt=None,
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
        tags_str="",
        ollama_models=ollama_models
    )


@app.route("/prompt/<int:prompt_id>/edit", methods=["GET", "POST"])
def edit_prompt(prompt_id: int):
    categories = get_categories()
    tools = get_tools()
    ollama_models = ollama_list_models()

    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not prompt:
        conn.close()
        flash("Prompt not found.")
        return redirect(url_for("index"))

    variants = conn.execute("SELECT id, title FROM prompts WHERE parent_id=?", (prompt_id,)).fetchall()
    parent = None
    if prompt["parent_id"]:
        parent = conn.execute("SELECT id, title FROM prompts WHERE id=?", (prompt["parent_id"],)).fetchone()
    
    current_tags = get_tags_for_prompt(conn, prompt_id)
    tags_str = ", ".join(current_tags)
    conn.close()

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        ensure_category_tool_exist(category, tool)

        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content, notes = request.form.get("content", "").strip(), request.form.get("notes", "").strip()
        tags_input = request.form.get("tags", "").strip()

        new_thumb = prompt["thumbnail"]
        if request.form.get("remove_thumbnail") == "on":
            delete_thumbnail_if_unused(new_thumb)
            new_thumb = None
        f = request.files.get("thumbnail")
        if f and f.filename and is_allowed_thumb(f.filename):
            if new_thumb: delete_thumbnail_if_unused(new_thumb)
            new_thumb = save_thumbnail(f)

        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            conn.execute(
                """
                UPDATE prompts
                SET title=?, category=?, tool=?, prompt_type=?, content=?, notes=?, thumbnail=?, updated_at=?
                WHERE id=?
                """,
                (title, category, tool, prompt_type, content, notes, new_thumb, now, prompt_id),
            )
            save_tags(conn, prompt_id, tags_input)
            conn.commit()
        return redirect(url_for("index"))

    return render_template(
        "edit_prompt.html",
        prompt=prompt,
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
        variants=variants,
        parent=parent,
        tags_str=tags_str,
        ollama_models=ollama_models
    )


@app.route("/prompt/<int:prompt_id>/duplicate", methods=["POST"])
def duplicate_prompt(prompt_id: int):
    conn = get_db()
    p = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()

    if not p:
        conn.close()
        flash("Prompt not found.")
        return redirect(url_for("index"))

    as_variant = request.form.get("as_variant") == "true"
    new_title = f"{p['title']} (Variant)" if as_variant else f"Copy of {p['title']}"
    parent_id = p["parent_id"] if (as_variant and p["parent_id"]) else (p["id"] if as_variant else None)

    new_thumb = None
    if p["thumbnail"]:
        new_thumb = copy_thumbnail(p["thumbnail"])

    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (new_title, p["category"], p["tool"], p["prompt_type"], p["content"], p["notes"], new_thumb, parent_id, now, now),
    )
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    tags = get_tags_for_prompt(conn, prompt_id)
    save_tags(conn, new_id, ", ".join(tags))
    
    conn.commit()
    conn.close()

    flash("Variant created." if as_variant else "Prompt duplicated.")
    return redirect(url_for("edit_prompt", prompt_id=new_id))


@app.route("/prompt/<int:prompt_id>/delete", methods=["POST"])
def delete_prompt(prompt_id: int):
    conn = get_db()
    row = conn.execute("SELECT thumbnail FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    thumb = row["thumbnail"] if row else None

    conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.execute("UPDATE prompts SET parent_id = NULL WHERE parent_id = ?", (prompt_id,))
    conn.commit()
    conn.close()

    if thumb:
        delete_thumbnail_if_unused(thumb)

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
        if request.form.get("final_override"):
            final_text = request.form.get("final_override").strip()

    return render_template(
        "render_prompt.html",
        prompt=prompt,
        placeholders=placeholders,
        final_text=final_text,
    )


# -------------------------
# Import / Export / Manage
# -------------------------
@app.route("/prompt/import", methods=["GET", "POST"])
def import_prompt():
    categories, tools = get_categories(), get_tools()

    if request.method == "POST":
        files = request.files.getlist("files")
        if not files or all((f is None or f.filename == "") for f in files):
            return redirect(url_for("import_prompt"))

        batch_cat = request.form.get("batch_category", "").strip()
        batch_tool = request.form.get("batch_tool", "").strip()
        
        # Helper to safely add cat/tool using CURRENT connection
        # (Fixes the "Database is locked" error by avoiding recursive connections)
        def safe_add_meta(c, t, active_conn):
            if c:
                try: active_conn.execute("INSERT INTO categories (name) VALUES (?)", (c,))
                except sqlite3.IntegrityError: pass
            if t:
                try: active_conn.execute("INSERT INTO tools (name) VALUES (?)", (t,))
                except sqlite3.IntegrityError: pass

        batch_type = request.form.get("batch_prompt_type", "Instruction").strip()
        batch_notes = request.form.get("batch_notes", "").strip()
        single_mode = request.form.get("single_prompt_per_file") == "on"
        use_first_line_as_title = request.form.get("use_first_line_as_title") == "on"
        prefer_metadata = request.form.get("prefer_file_metadata") == "on"

        conn = get_db()
        # Initialize defaults safely
        safe_add_meta(batch_cat, batch_tool, conn)
        
        now = datetime.utcnow().isoformat()

        for f in files:
            if not f or not f.filename: continue
            filename = secure_filename(f.filename)
            if not filename.lower().endswith(".txt"): continue

            raw = f.read().decode("utf-8", errors="ignore")
            chunks = [raw] if single_mode else re.split(r'\n\s*\n', raw)

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk: continue
                p = parse_prompt_txt(chunk)
                
                title = filename.rsplit(".", 1)[0]
                if p["title"]: title = p["title"]
                elif use_first_line_as_title:
                    lines = p["content"].split('\n', 1)
                    if lines[0].strip():
                        title = lines[0].strip()
                        p["content"] = lines[1].strip() if len(lines) > 1 else ""
                
                cat = "Image"
                if prefer_metadata and p["category"]: cat = p["category"]
                elif batch_cat: cat = batch_cat
                elif p["category"]: cat = p["category"]
                
                tool = "Generic"
                if prefer_metadata and p["tool"]: tool = p["tool"]
                elif batch_tool: tool = batch_tool
                elif p["tool"]: tool = p["tool"]

                # Add potential new metadata (safely)
                safe_add_meta(cat, tool, conn)

                ptype = batch_type
                if p["prompt_type"] and (prefer_metadata or not batch_type):
                    ptype = p["prompt_type"]

                pnotes = p["notes"]
                if batch_notes:
                    pnotes = (batch_notes + "\n\n" + pnotes).strip()

                conn.execute(
                    """
                    INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (title, cat, tool, ptype, p["content"], pnotes, None, now, now),
                )

        conn.commit()
        conn.close()
        flash("Import successful.")
        return redirect(url_for("index"))

    return render_template(
        "import_prompt.html",
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
    )

@app.route("/prompt/import/preview", methods=["POST"])
def import_prompt_preview():
    categories, tools = get_categories(), get_tools()
    f = request.files.get("file")
    if not f: return redirect(url_for("import_prompt"))
    
    raw = f.read().decode("utf-8", errors="ignore")
    p = parse_prompt_txt(raw)
    title = p["title"] or secure_filename(f.filename).rsplit(".", 1)[0]
    
    prompt_like = {
        "title": title,
        "category": p["category"] or "Image",
        "tool": p["tool"] or "Generic",
        "prompt_type": p["prompt_type"] or "Instruction",
        "content": p["content"],
        "notes": p["notes"],
        "thumbnail": None
    }

    return render_template(
        "edit_prompt.html",
        prompt=prompt_like,
        categories=categories,
        tools=tools,
        prompt_types=PROMPT_TYPES,
        imported_preview=True,
        tags_str=""
    )


@app.route("/manage", methods=["GET"])
def manage():
    return render_template("manage.html", categories=get_categories(), tools=get_tools())

@app.route("/manage/category/add", methods=["POST"])
def add_category():
    n = request.form.get("name", "").strip()
    if n: ensure_category_tool_exist(n, None)
    return redirect(url_for("manage"))

@app.route("/manage/category/delete", methods=["POST"])
def delete_category():
    n = request.form.get("name", "").strip()
    if n:
        with get_db() as conn: conn.execute("DELETE FROM categories WHERE name=?", (n,)); conn.commit()
    return redirect(url_for("manage"))

@app.route("/manage/tool/add", methods=["POST"])
def add_tool():
    n = request.form.get("name", "").strip()
    if n: ensure_category_tool_exist(None, n)
    return redirect(url_for("manage"))

@app.route("/manage/tool/delete", methods=["POST"])
def delete_tool():
    n = request.form.get("name", "").strip()
    if n:
        with get_db() as conn: conn.execute("DELETE FROM tools WHERE name=?", (n,)); conn.commit()
    return redirect(url_for("manage"))

@app.route("/bulk_update", methods=["POST"])
def bulk_update():
    ids = request.form.getlist("prompt_ids")
    cat = request.form.get("bulk_category")
    tool = request.form.get("bulk_tool")
    ptype = request.form.get("bulk_prompt_type")
    tags_str = request.form.get("bulk_tags", "").strip()
    
    ensure_category_tool_exist(cat, tool)

    if ids:
        with get_db() as conn:
            for pid in ids:
                now_str = datetime.utcnow().isoformat()
                if cat: conn.execute("UPDATE prompts SET category=?, updated_at=? WHERE id=?", (cat, now_str, pid))
                if tool: conn.execute("UPDATE prompts SET tool=?, updated_at=? WHERE id=?", (tool, now_str, pid))
                if ptype: conn.execute("UPDATE prompts SET prompt_type=?, updated_at=? WHERE id=?", (ptype, now_str, pid))
                if tags_str:
                    raw_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                    for tag_name in raw_tags:
                        try: conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                        except sqlite3.IntegrityError: pass
                        tag_id = conn.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()[0]
                        try: conn.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (int(pid), tag_id))
                        except sqlite3.IntegrityError: pass
                    conn.execute("UPDATE prompts SET updated_at=? WHERE id=?", (now_str, pid))
            conn.commit()
    return redirect(url_for("index"))


# -------------------------
# Backup / Restore
# -------------------------
@app.route("/backup/download")
def backup_download():
    init_db()
    if not DB_PATH.exists():
        flash("Database file not found.")
        return redirect(url_for("manage"))
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buffer = io.BytesIO()

    # Create ZIP archive (DB + Thumbs)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DB_PATH, arcname="prompts.db")
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and f.name != ".gitkeep":
                    zf.write(f, arcname=f"thumbs/{f.name}")
    
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"prompts_backup_{ts}.zip"
    )

@app.route("/backup/restore", methods=["GET", "POST"])
def restore_backup():
    init_db()
    if request.method == "GET":
        return render_template("restore.html")
    f = request.files.get("db_file")
    if not f or f.filename.strip() == "":
        flash("No file selected.")
        return redirect(url_for("restore_backup"))
    
    filename = f.filename.lower()
    if filename.endswith(".zip"):
        try:
            with zipfile.ZipFile(f, 'r') as zf:
                if "prompts.db" in zf.namelist():
                    with zf.open("prompts.db") as source, open(DB_PATH, "wb") as target:
                        shutil.copyfileobj(source, target)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                for member in zf.namelist():
                    if member.startswith("thumbs/") and not member.endswith("/"):
                        target_path = UPLOAD_DIR / os.path.basename(member)
                        with zf.open(member) as source, open(target_path, "wb") as target:
                            shutil.copyfileobj(source, target)
            flash("Backup restored."); return redirect(url_for("manage"))
        except: flash("Restore failed."); return redirect(url_for("restore_backup"))
    elif filename.endswith(".db") or filename.endswith(".sqlite"):
        try:
            f.save(DB_PATH)
            flash("Database restored."); return redirect(url_for("manage"))
        except: pass
    
    return redirect(url_for("restore_backup"))

# ---------------------------
# Ollama Integration
# ---------------------------
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_list_models():
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

def ollama_generate(prompt: str, model: str | None = None, system: str | None = None, images: list[str] | None = None) -> str:
    """
    Unified generic generator. 
    Supports vision if 'images' (list of base64 strings) is provided.
    """
    payload = {
        "model": model or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system: payload["system"] = system
    if images: payload["images"] = images  # Ollama expects list of base64 strings

    # Increased timeout to 120s to account for slower vision models
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

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

    # --- POST Handling ---
    if not ollama_available:
        flash("Ollama not available.")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    session["ollama_model"] = model
    instruction = (request.form.get("instruction") or "").strip()
    mode = request.form.get("mode") or "refine"

    # Grab content from FORM (allows editing source before refining)
    base = request.form.get("content") or prompt["content"]

    try:
        refined = ollama_generate(f"MODE: {mode}\nINSTRUCTION: {instruction}\n\nPROMPT:\n{base}", model=model, system="Output only refined prompt.")
    except Exception as e:
        flash(f"Ollama error: {e}")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    # --- SAVE LOGIC ---
    if request.form.get("save_as_new") == "on":
        # 1. Save as New Variant
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.form.get("new_title") or f"{prompt['title']} (Refined)",
                    prompt["category"],
                    prompt["tool"],
                    prompt["prompt_type"],
                    refined,
                    prompt["notes"],
                    datetime.utcnow().isoformat(),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        flash("Refined prompt saved as new.")
        return redirect(url_for("index"))

    elif request.form.get("overwrite") == "on":
        # 2. Explicit Overwrite
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE prompts SET content = ?, updated_at = ? WHERE id = ?", (refined, datetime.utcnow().isoformat(), prompt_id))
            conn.commit()
        flash("Prompt updated.")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    else:
        # 3. Preview Mode (No DB Save)
        flash("Refinement generated (Unsaved). Check a box to save.")
        
        # Create a temporary dict to show the result in the UI
        p_preview = dict(prompt)
        p_preview['content'] = refined
        
        return render_template("refine_prompt.html", prompt=p_preview, models=models, ollama_available=ollama_available)

@app.route("/refine_draft", methods=["POST"])
def refine_draft():
    init_db()
    models = ollama_list_models()
    
    categories = get_categories()
    tools = get_tools()
    base = request.form.get("content") or ""
    
    prompt_like = {
        "id": None,
        "title": request.form.get("title", ""),
        "category": request.form.get("category", categories[0] if categories else "Other"),
        "tool": request.form.get("tool", tools[0] if tools else "Other"),
        "prompt_type": request.form.get("prompt_type", "Instruction"),
        "content": base,
        "notes": request.form.get("notes", ""),
    }
    
    if not models:
        return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refine_error="Ollama unavailable", ollama_available=False, tags_str="")

    model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    session["ollama_model"] = model
    try:
        refined = ollama_generate(f"MODE: {request.form.get('mode')}\nINSTRUCTION: {request.form.get('instruction')}\n\nPROMPT:\n{base}", model=model, system="Output only refined prompt.")
        return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refined_draft=refined, ollama_available=True, tags_str="")
    except Exception as e:
        return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refine_error=str(e), ollama_available=True, tags_str="")

@app.route("/api/generate_draft_from_image", methods=["POST"])
def generate_draft_from_image():
    # 1. Inputs
    f = request.files.get("image")
    vision_model = request.form.get("vision_model")
    text_model = request.form.get("text_model")
    custom_vision_instruction = request.form.get("custom_vision_instruction", "").strip()
    custom_draft_instruction = request.form.get("custom_draft_instruction", "").strip()
    
    if not f or not vision_model or not text_model:
        return jsonify({"error": "Missing image or model selection."}), 400

    # 2. Process Image (Save & Encode)
    try:
        # A) Save file to disk immediately (so it can be used as a thumbnail)
        thumbnail_path = save_thumbnail(f)
        
        # B) Read file back to encode for Ollama (since file pointer moved)
        # We read from the saved file on disk
        saved_file_path = BASE_DIR / "static" / thumbnail_path
        with open(saved_file_path, "rb") as image_file:
            img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    # 3. Vision Step
    vision_sys = """You are a visual analysis model.
Carefully describe the contents of the provided image.
Focus only on what is directly visible.
Structure your response clearly using short sections or bullet points where appropriate.
Include: Subject, Appearance, Environment, Composition, Lighting, Mood, Visual Style.
Output should be factual and observational."""

    vision_prompt = "Describe this image."
    if custom_vision_instruction:
        vision_prompt += f" {custom_vision_instruction}"

    try:
        vision_response = ollama_generate(
            prompt=vision_prompt, 
            model=vision_model, 
            system=vision_sys, 
            images=[img_b64]
        )
    except Exception as e:
        return jsonify({"error": f"Vision model failed: {str(e)}"}), 500

    # 4. Drafting Step
    draft_sys = """You are assisting with AI prompt creation.
Based on the following visual description, generate a draft AI image prompt suitable for use with modern image generation models (e.g. Flux, Stable Diffusion).
Rules:
Do not invent details not present in the description.
Use neutral, reusable phrasing.
Prefer descriptive language.
The result should be a starting point, not a finished prompt.
Output a single paragraph prompt."""

    draft_user_prompt = f"Visual description:\n{vision_response}"
    if custom_draft_instruction:
        draft_user_prompt += f"\n\nAdditional Instructions:\n{custom_draft_instruction}"

    try:
        draft_response = ollama_generate(
            prompt=draft_user_prompt,
            model=text_model,
            system=draft_sys
        )
    except Exception as e:
        return jsonify({"error": f"Text model failed: {str(e)}"}), 500

    return jsonify({
        "success": True, 
        "description": vision_response, 
        "draft_prompt": draft_response,
        "thumbnail_path": thumbnail_path  # Return the path so frontend can use it
    })


# Raw API
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


if __name__ == "__main__":
    init_db()
    app.run(debug=True)