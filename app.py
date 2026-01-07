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
from datetime import datetime
from pathlib import Path
from PIL import Image  # pip install Pillow

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
    
    # Core Tables
    conn.execute("CREATE TABLE IF NOT EXISTS categories (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    conn.execute("CREATE TABLE IF NOT EXISTS tools (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    
    # Prompts Table
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

    # Saved Views Table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS saved_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            query_params TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Tagging Tables
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

    # Migrations
    if not column_exists(conn, "prompts", "thumbnail"):
        conn.execute("ALTER TABLE prompts ADD COLUMN thumbnail TEXT")
    if not column_exists(conn, "prompts", "parent_id"):
        conn.execute("ALTER TABLE prompts ADD COLUMN parent_id INTEGER")

    # Seed Defaults
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
    conn = get_db()
    if category:
        try: conn.execute("INSERT INTO categories (name) VALUES (?)", (category,))
        except: pass
    if tool:
        try: conn.execute("INSERT INTO tools (name) VALUES (?)", (tool,))
        except: pass
    conn.commit(); conn.close()

def is_allowed_thumb(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_THUMB_EXTS

def save_thumbnail(file_storage) -> str:
    original_filename = secure_filename(file_storage.filename)
    ext = Path(original_filename).suffix.lower()
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / new_name

    try:
        img = Image.open(file_storage)
        if ext == '.gif':
            file_storage.seek(0); file_storage.save(dest)
        else:
            img.thumbnail((600, 600))
            if ext in ['.jpg', '.jpeg']:
                if img.mode != 'RGB': img = img.convert('RGB')
                img.save(dest, optimize=True, quality=85)
            elif ext == '.png': img.save(dest, optimize=True)
            elif ext == '.webp': img.save(dest, optimize=True, quality=85)
            else: img.save(dest)
        return f"uploads/thumbs/{new_name}"
    except Exception as e:
        print(f"Image processing failed: {e}")
        file_storage.seek(0); file_storage.save(dest)
        return f"uploads/thumbs/{new_name}"

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
        s = line.strip()
        if s == "---": content_start = i + 1; break
        if s.startswith("#") and ":" in s:
            try: k, v = s[1:].split(":", 1); meta[k.strip().lower()] = v.strip(); content_start = i + 1
            except: pass
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
    sort_by = request.args.get("sort_by", "updated_at")
    sort_dir = request.args.get("sort_dir", "desc")
    try: page = max(1, int(request.args.get("page", 1)))
    except: page = 1

    ALLOWED_SORTS = {"updated_at", "created_at", "title", "category", "tool", "prompt_type"}
    if sort_by not in ALLOWED_SORTS: sort_by = "updated_at"
    if sort_dir not in ["asc", "desc"]: sort_dir = "desc"

    conn = get_db()
    
    base_query = " FROM prompts p"
    if tag_filter != "all":
        base_query += " JOIN prompt_tags pt ON p.id = pt.prompt_id JOIN tags t ON pt.tag_id = t.id"
    
    base_query += " WHERE 1=1"
    params = []

    if category != "all": base_query += " AND p.category = ?"; params.append(category)
    if tool != "all": base_query += " AND p.tool = ?"; params.append(tool)
    if tag_filter != "all": base_query += " AND t.name = ?"; params.append(tag_filter)
    
    if q:
        base_query += " AND (LOWER(p.title) LIKE ? OR LOWER(p.content) LIKE ? OR LOWER(COALESCE(p.notes, '')) LIKE ?)"
        like = f"%{q.lower()}%"; params.extend([like, like, like])

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
    if not f or not is_allowed_thumb(f.filename): return jsonify({"error": "Invalid file type"}), 400
    conn = get_db()
    row = conn.execute("SELECT thumbnail FROM prompts WHERE id=?", (prompt_id,)).fetchone()
    if not row: conn.close(); return jsonify({"error": "Prompt not found"}), 404
    old_thumb = row["thumbnail"]
    new_path = save_thumbnail(f)
    if old_thumb: delete_thumbnail_if_unused(old_thumb)
    conn.execute("UPDATE prompts SET thumbnail=?, updated_at=? WHERE id=?", (new_path, datetime.utcnow().isoformat(), prompt_id))
    conn.commit(); conn.close()
    return jsonify({"success": True, "thumbnail_url": url_for('static', filename=new_path)})


@app.route("/view/save", methods=["POST"])
def save_view():
    name, qs = request.form.get("view_name", "").strip(), request.form.get("query_string", "")
    if not name or not qs: return redirect(url_for("index"))
    conn = get_db()
    conn.execute("INSERT INTO saved_views (name, query_params, created_at) VALUES (?, ?, ?)", (name, qs, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    flash(f"View '{name}' saved."); return redirect(f"/?{qs}")

@app.route("/view/delete/<int:view_id>", methods=["POST"])
def delete_view(view_id):
    conn = get_db()
    conn.execute("DELETE FROM saved_views WHERE id=?", (view_id,))
    conn.commit(); conn.close()
    return redirect(url_for("index"))


@app.route("/prompt/new", methods=["GET", "POST"])
def new_prompt():
    categories, tools = get_categories(), get_tools()
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        ensure_category_tool_exist(category, tool)

        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content, notes = request.form.get("content", "").strip(), request.form.get("notes", "").strip()
        tags_str = request.form.get("tags", "").strip()

        if not title or not content: return redirect(url_for("new_prompt"))

        thumb_path = None
        f = request.files.get("thumbnail")
        if f and f.filename and is_allowed_thumb(f.filename): thumb_path = save_thumbnail(f)

        now = datetime.utcnow().isoformat()
        conn = get_db()
        conn.execute("INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                     (title, category, tool, prompt_type, content, notes, thumb_path, now, now))
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        save_tags(conn, new_id, tags_str)
        conn.commit(); conn.close()
        return redirect(url_for("index"))
    return render_template("edit_prompt.html", prompt=None, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, tags_str="")


@app.route("/prompt/<int:prompt_id>/edit", methods=["GET", "POST"])
def edit_prompt(prompt_id: int):
    categories = get_categories()
    tools = get_tools()
    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not prompt: conn.close(); return redirect(url_for("index"))

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
            delete_thumbnail_if_unused(new_thumb); new_thumb = None
        f = request.files.get("thumbnail")
        if f and f.filename and is_allowed_thumb(f.filename):
            if new_thumb: delete_thumbnail_if_unused(new_thumb)
            new_thumb = save_thumbnail(f)

        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            conn.execute("UPDATE prompts SET title=?, category=?, tool=?, prompt_type=?, content=?, notes=?, thumbnail=?, updated_at=? WHERE id=?",
                         (title, category, tool, prompt_type, content, notes, new_thumb, now, prompt_id))
            save_tags(conn, prompt_id, tags_input)
            conn.commit()
        return redirect(url_for("index"))

    return render_template("edit_prompt.html", prompt=prompt, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, variants=variants, parent=parent, tags_str=tags_str)


@app.route("/prompt/<int:prompt_id>/duplicate", methods=["POST"])
def duplicate_prompt(prompt_id: int):
    conn = get_db()
    p = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    if not p: conn.close(); return redirect(url_for("index"))

    as_variant = request.form.get("as_variant") == "true"
    new_title = f"{p['title']} (Variant)" if as_variant else f"Copy of {p['title']}"
    parent_id = p["parent_id"] if (as_variant and p["parent_id"]) else (p["id"] if as_variant else None)

    new_thumb = None
    if p["thumbnail"]: new_thumb = copy_thumbnail(p["thumbnail"])

    now = datetime.utcnow().isoformat()
    conn.execute("INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (new_title, p["category"], p["tool"], p["prompt_type"], p["content"], p["notes"], new_thumb, parent_id, now, now))
    new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    
    tags = get_tags_for_prompt(conn, prompt_id)
    save_tags(conn, new_id, ", ".join(tags))
    
    conn.commit(); conn.close()
    flash("Variant created." if as_variant else "Prompt duplicated.")
    return redirect(url_for("edit_prompt", prompt_id=new_id))


@app.route("/prompt/<int:prompt_id>/delete", methods=["POST"])
def delete_prompt(prompt_id: int):
    conn = get_db()
    row = conn.execute("SELECT thumbnail FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    thumb = row["thumbnail"] if row else None
    conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))
    conn.execute("UPDATE prompts SET parent_id = NULL WHERE parent_id = ?", (prompt_id,))
    conn.commit(); conn.close()
    if thumb: delete_thumbnail_if_unused(thumb)
    flash("Prompt deleted."); return redirect(url_for("index"))


@app.route("/prompt/<int:prompt_id>/render", methods=["GET", "POST"])
def render_prompt(prompt_id: int):
    conn = get_db()
    prompt = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
    conn.close()
    if not prompt: return redirect(url_for("index"))

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

    return render_template("render_prompt.html", prompt=prompt, placeholders=placeholders, final_text=final_text)


@app.route("/prompt/import", methods=["GET", "POST"])
def import_prompt():
    categories, tools = get_categories(), get_tools()
    if request.method == "POST":
        files = request.files.getlist("files")
        if not files: return redirect(url_for("import_prompt"))

        batch_cat, batch_tool = request.form.get("batch_category", ""), request.form.get("batch_tool", "")
        ensure_category_tool_exist(batch_cat, batch_tool)
        batch_type = request.form.get("batch_prompt_type", "Instruction")
        batch_notes = request.form.get("batch_notes", "")
        single_mode = request.form.get("single_prompt_per_file") == "on"
        use_first_line_as_title = request.form.get("use_first_line_as_title") == "on"
        prefer_metadata = request.form.get("prefer_file_metadata") == "on"

        conn = get_db(); now = datetime.utcnow().isoformat()
        for f in files:
            if not f or not f.filename.endswith(".txt"): continue
            raw = f.read().decode("utf-8", errors="ignore")
            chunks = [raw] if single_mode else re.split(r'\n\s*\n', raw)
            for chunk in chunks:
                if not chunk.strip(): continue
                p = parse_prompt_txt(chunk)
                title = secure_filename(f.filename).rsplit(".", 1)[0]
                if p["title"]: title = p["title"]
                elif use_first_line_as_title:
                    lines = p["content"].split('\n', 1)
                    if lines[0].strip():
                        title = lines[0].strip()
                        p["content"] = lines[1].strip() if len(lines) > 1 else ""
                
                cat = "Image"
                if prefer_metadata and p["category"]: cat = p["category"]
                elif batch_cat: cat = batch_cat
                
                tool = "Generic"
                if prefer_metadata and p["tool"]: tool = p["tool"]
                elif batch_tool: tool = batch_tool
                
                ensure_category_tool_exist(cat, tool)
                ptype = batch_type
                if p["prompt_type"] and (prefer_metadata or not batch_type): ptype = p["prompt_type"]
                pnotes = p["notes"]
                if batch_notes: pnotes = (batch_notes + "\n\n" + pnotes).strip()

                conn.execute("INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                             (title, cat, tool, ptype, p["content"], pnotes, now, now))
        conn.commit(); conn.close()
        flash("Import successful."); return redirect(url_for("index"))
    return render_template("import_prompt.html", categories=categories, tools=tools, prompt_types=PROMPT_TYPES)

@app.route("/prompt/import/preview", methods=["POST"])
def import_prompt_preview():
    categories, tools = get_categories(), get_tools()
    f = request.files.get("file")
    if not f: return redirect(url_for("import_prompt"))
    raw = f.read().decode("utf-8", errors="ignore")
    p = parse_prompt_txt(raw)
    prompt_like = {
        "title": p["title"] or secure_filename(f.filename).rsplit(".", 1)[0],
        "category": p["category"] or "Image",
        "tool": p["tool"] or "Generic",
        "prompt_type": p["prompt_type"] or "Instruction",
        "content": p["content"],
        "notes": p["notes"],
        "thumbnail": None
    }
    return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, imported_preview=True, tags_str="")

@app.route("/manage", methods=["GET"])
def manage(): return render_template("manage.html", categories=get_categories(), tools=get_tools())

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
    cat, tool, ptype = request.form.get("bulk_category"), request.form.get("bulk_tool"), request.form.get("bulk_prompt_type")
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

@app.route("/backup/download")
def backup_download():
    init_db()
    if not DB_PATH.exists(): return redirect(url_for("manage"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DB_PATH, arcname="prompts.db")
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file() and f.name != ".gitkeep": zf.write(f, arcname=f"thumbs/{f.name}")
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=f"prompts_backup_{ts}.zip")

@app.route("/backup/restore", methods=["GET", "POST"])
def restore_backup():
    init_db()
    if request.method == "GET": return render_template("restore.html")
    f = request.files.get("db_file")
    if not f or f.filename.strip() == "": return redirect(url_for("restore_backup"))
    if f.filename.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(f, 'r') as zf:
                if "prompts.db" in zf.namelist():
                    with zf.open("prompts.db") as source, open(DB_PATH, "wb") as target: shutil.copyfileobj(source, target)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                for member in zf.namelist():
                    if member.startswith("thumbs/") and not member.endswith("/"):
                        target_path = UPLOAD_DIR / os.path.basename(member)
                        with zf.open(member) as source, open(target_path, "wb") as target: shutil.copyfileobj(source, target)
            flash("Backup restored."); return redirect(url_for("manage"))
        except: flash("Restore failed."); return redirect(url_for("restore_backup"))
    elif f.filename.lower().endswith((".db", ".sqlite")):
        f.save(DB_PATH)
        flash("Database restored."); return redirect(url_for("manage"))
    return redirect(url_for("restore_backup"))

@app.route("/prompt/<int:prompt_id>/raw", methods=["GET"])
def prompt_raw(prompt_id):
    p = get_prompt(prompt_id)
    if not p: return jsonify({"error": "not found"}), 404
    return jsonify({"id": p["id"], "title": p["title"], "content": p["content"]})

# Ollama
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def ollama_list_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200: return [m["name"] for m in r.json().get("models", [])]
    except: pass
    return []

def ollama_generate(prompt, model=None, system=None):
    payload = {"model": model or OLLAMA_DEFAULT_MODEL, "prompt": prompt, "stream": False}
    if system: payload["system"] = system
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("response", "").strip()

@app.route("/prompt/<int:prompt_id>/refine", methods=["GET", "POST"])
def refine_prompt(prompt_id):
    prompt = get_prompt(prompt_id)
    if not prompt: return redirect(url_for("index"))
    models = ollama_list_models()
    if request.method == "GET": return render_template("refine_prompt.html", prompt=prompt, models=models, ollama_available=bool(models))
    if not models: return redirect(url_for("render_prompt", prompt_id=prompt_id))
    
    model = request.form.get("model")
    session["ollama_model"] = model
    try:
        refined = ollama_generate(f"MODE: {request.form.get('mode')}\nINSTRUCTION: {request.form.get('instruction')}\n\nPROMPT:\n{prompt['content']}", model=model, system="Refine prompt.")
    except Exception as e:
        flash(f"Ollama error: {e}"); return redirect(url_for("render_prompt", prompt_id=prompt_id))

    if request.form.get("save_as_new"):
        with get_db() as conn:
            conn.execute("INSERT INTO prompts (title, category, tool, prompt_type, content, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                         (f"{prompt['title']} (Refined)", prompt["category"], prompt["tool"], prompt["prompt_type"], refined, prompt["notes"], datetime.utcnow().isoformat(), datetime.utcnow().isoformat()))
            conn.commit()
        return redirect(url_for("index"))
    else:
        with get_db() as conn:
            conn.execute("UPDATE prompts SET content=?, updated_at=? WHERE id=?", (refined, datetime.utcnow().isoformat(), prompt_id))
            conn.commit()
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

@app.route("/refine_draft", methods=["POST"])
def refine_draft():
    models = ollama_list_models()
    categories, tools = get_categories(), get_tools()
    base = request.form.get("content", "")
    p_like = {"title": request.form.get("title"), "category": "Other", "tool": "Generic", "prompt_type": "Instruction", "content": base, "notes": ""}
    
    if not models: return render_template("edit_prompt.html", prompt=p_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refine_error="Ollama unavailable", ollama_available=False, tags_str="")
    
    try:
        refined = ollama_generate(f"MODE: {request.form.get('mode')}\nINSTRUCTION: {request.form.get('instruction')}\n\nPROMPT:\n{base}", model=request.form.get("model"), system="Refine prompt.")
        return render_template("edit_prompt.html", prompt=p_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refined_draft=refined, ollama_available=True, tags_str="")
    except Exception as e:
        return render_template("edit_prompt.html", prompt=p_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refine_error=str(e), ollama_available=True, tags_str="")

if __name__ == "__main__":
    init_db()
    app.run(debug=True)