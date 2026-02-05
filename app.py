import sqlite3
import re
import uuid
import shutil
import requests
import zipfile
import io
import os
import math
import time
import json
import base64
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageSequence

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session, jsonify
from werkzeug.utils import secure_filename


# -------------------------
# Descriptor token injection
# -------------------------
DESCRIPTOR_TOKEN_RE = re.compile(r"\{\{\s*descriptor\s*:\s*(\d+)\s*\}\}")

def inject_descriptors(text_value: str) -> str:
    """Replace {{descriptor:ID}} tokens with the descriptor's rendered_text.
    Safe: if missing, leaves a visible placeholder.
    """
    if not text_value:
        return text_value

    ids = [m.group(1) for m in DESCRIPTOR_TOKEN_RE.finditer(text_value)]
    if not ids:
        return text_value

    # De-duplicate while keeping order
    seen = set()
    unique_ids = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            unique_ids.append(i)

    db = get_db()
    qmarks = ",".join(["?"] * len(unique_ids))
    rows = db.execute(
        f"SELECT id, rendered_text FROM descriptors WHERE id IN ({qmarks})",
        [int(x) for x in unique_ids],
    ).fetchall()
    mapping = {str(r["id"]): (r["rendered_text"] or "") for r in rows}

    def repl(m):
        did = m.group(1)
        return mapping.get(did, f"[Missing descriptor:{did}]")

    return DESCRIPTOR_TOKEN_RE.sub(repl, text_value)

app = Flask(__name__)
app.secret_key = "change-me-to-something-random"

@app.context_processor
def inject_global_flags():
    # Keep nav/visibility logic out of templates as much as possible.
    # ollama_ready is a hard gate for Remix.
    return {
        "nsfw_unlocked": nsfw_is_unlocked(),
        "ollama_ready": ollama_is_ready_cached(),
        "ollama_models": ollama_models_cached(),
    }
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

PROMPT_TYPES = ["Generation", "Edit", "Instruction"]
UPLOAD_DIR = BASE_DIR / "static" / "uploads" / "thumbs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
ITEMS_PER_PAGE = 12

# -------------------------
# NSFW Lock / Visibility
# -------------------------
NSFW_LOCK_ENABLED_DEFAULT = True  # NSFW hidden unless explicitly unlocked (per session)

# NOTE: Do NOT include "adult" here (too generic / used in SFW contexts too).
NSFW_TAGS = {"nsfw", "porn", "explicit", "18+", "xxx"}

def nsfw_is_unlocked() -> bool:
    """
    Returns True if this session is allowed to view NSFW prompts.
    Default: locked (False).
    """
    if not NSFW_LOCK_ENABLED_DEFAULT:
        return True
    return bool(session.get("show_nsfw", False))

def apply_nsfw_filter(base_query: str, params: list) -> tuple[str, list]:
    """
    Adds a WHERE clause to hide NSFW-tagged prompts unless session is unlocked.
    Assumes main prompt table alias is 'p' (as in your library queries).
    """
    if nsfw_is_unlocked():
        return base_query, params

    placeholders = ",".join(["?"] * len(NSFW_TAGS))
    base_query += f"""
        AND NOT EXISTS (
            SELECT 1
            FROM prompt_tags nspt
            JOIN tags nst ON nst.id = nspt.tag_id
            WHERE nspt.prompt_id = p.id
              AND LOWER(nst.name) IN ({placeholders})
        )
    """
    params = params + [t.lower() for t in NSFW_TAGS]
    return base_query, params

def prompt_is_nsfw(conn: sqlite3.Connection, prompt_id: int) -> bool:
    """
    True if the prompt has a tag that matches NSFW_TAGS (case-insensitive).
    """
    placeholders = ",".join(["?"] * len(NSFW_TAGS))
    row = conn.execute(
        f"""
        SELECT 1
        FROM prompt_tags pt
        JOIN tags t ON t.id = pt.tag_id
        WHERE pt.prompt_id = ?
          AND LOWER(t.name) IN ({placeholders})
        LIMIT 1
        """,
        [prompt_id, *[t.lower() for t in NSFW_TAGS]],
    ).fetchone()
    return bool(row)


# -------------------------
# DB helpers
# -------------------------
def get_db(path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # ensure FK constraints (for ON DELETE CASCADE etc.)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
    except Exception:
        pass
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
            group_id INTEGER,
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

    
    # Descriptor Packs (generic: characters, creatures, environments, props, etc.)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS descriptor_packs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pack_type TEXT NOT NULL DEFAULT 'generic',
            pack_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Saved descriptor instances generated from a pack
    conn.execute("""
        CREATE TABLE IF NOT EXISTS descriptors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            pack_id INTEGER NOT NULL,
            pack_type TEXT NOT NULL DEFAULT 'generic',
            values_json TEXT NOT NULL,
            rendered_text TEXT NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(pack_id) REFERENCES descriptor_packs(id) ON DELETE CASCADE
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

    # -------------------------
    # Prompt Groups (theme / concept grouping)
    # -------------------------
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL
        )
    """)

    if not column_exists(conn, "prompts", "group_id"):
        conn.execute("ALTER TABLE prompts ADD COLUMN group_id INTEGER")

    if not column_exists(conn, "prompts", "thumbnail"):
        conn.execute("ALTER TABLE prompts ADD COLUMN thumbnail TEXT")
    if not column_exists(conn, "prompts", "parent_id"):
        conn.execute("ALTER TABLE prompts ADD COLUMN parent_id INTEGER")

    if conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0] == 0:
        for c in ["Image", "Video", "Music", "Other"]:
            conn.execute("INSERT INTO categories (name) VALUES (?)", (c,))
    if conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0] == 0:
        for t in ["Generic", "Flux", "Flux Kontext", "Qwen Edit", "Nano Banana", "Z-Image", "Suno", "RVC", "Other"]:
            conn.execute("INSERT INTO tools (name) VALUES (?)", (t,))

    
    # Prompt usage history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prompt_uses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT,
            FOREIGN KEY(prompt_id) REFERENCES prompts(id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_prompt_uses_prompt_id_created_at
        ON prompt_uses(prompt_id, created_at DESC)
    """)

#conn.commit()
 #   conn.close()

def get_categories() -> list[str]:
    with get_db() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM categories ORDER BY name COLLATE NOCASE")]

def get_tools() -> list[str]:
    with get_db() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM tools ORDER BY name COLLATE NOCASE")]

def get_all_tags() -> list[str]:
    with get_db() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM tags ORDER BY name ASC")]

def get_groups():
    with get_db() as conn:
        return conn.execute("SELECT * FROM prompt_groups ORDER BY name COLLATE NOCASE").fetchall()

def get_group(group_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM prompt_groups WHERE id = ?", (group_id,)).fetchone()

def ensure_group_exists(name: str, description: str | None = None) -> int | None:
    name = (name or "").strip()
    if not name:
        return None
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO prompt_groups (name, description, created_at) VALUES (?, ?, ?)",
                (name, (description or None), now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass
        row = conn.execute("SELECT id FROM prompt_groups WHERE name = ?", (name,)).fetchone()
        return int(row["id"]) if row else None

def get_saved_views():
    with get_db() as conn:
        return conn.execute("SELECT * FROM saved_views ORDER BY name ASC").fetchall()

def get_prompt(prompt_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()


# -------------------------
# Tag Logic
# -------------------------
def save_tags(conn, prompt_id, tags_str):
    conn.execute("DELETE FROM prompt_tags WHERE prompt_id=?", (prompt_id,))
    if not tags_str:
        return

    raw_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
    for tag_name in raw_tags:
        try:
            conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        except sqlite3.IntegrityError:
            pass

        tag_row = conn.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()
        if tag_row:
            try:
                conn.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (prompt_id, tag_row[0]))
            except sqlite3.IntegrityError:
                pass

def get_tags_for_prompt(conn, prompt_id):
    rows = conn.execute("""
        SELECT t.name FROM tags t
        JOIN prompt_tags pt ON t.id = pt.tag_id
        WHERE pt.prompt_id = ?
        ORDER BY t.name ASC
    """, (prompt_id,)).fetchall()
    return [r["name"] for r in rows]

def get_tags_for_prompts(conn, prompt_ids):
    """Fetch tags for many prompts in one query. Returns {prompt_id: [tag, ...]}"""
    if not prompt_ids:
        return {}

    placeholders = ",".join(["?"] * len(prompt_ids))
    rows = conn.execute(f"""
        SELECT pt.prompt_id, t.name
        FROM prompt_tags pt
        JOIN tags t ON t.id = pt.tag_id
        WHERE pt.prompt_id IN ({placeholders})
        ORDER BY t.name ASC
    """, prompt_ids).fetchall()

    out = {}
    for r in rows:
        out.setdefault(r["prompt_id"], []).append(r["name"])
    return out

def get_group_name_map(conn: sqlite3.Connection, group_ids: list[int]):
    """Returns {group_id: group_name} for given ids."""
    group_ids = [int(g) for g in group_ids if g is not None]
    if not group_ids:
        return {}
    placeholders = ",".join(["?"] * len(group_ids))
    rows = conn.execute(
        f"SELECT id, name FROM prompt_groups WHERE id IN ({placeholders})",
        group_ids,
    ).fetchall()
    return {int(r["id"]): r["name"] for r in rows}


# -------------------------
# Logic Helpers
# -------------------------
def ensure_category_tool_exist(category, tool):
    # Only opens a connection if it's called outside a transaction loop
    try:
        conn = get_db()
        if category:
            try:
                conn.execute("INSERT INTO categories (name) VALUES (?)", (category,))
            except:
                pass
        if tool:
            try:
                conn.execute("INSERT INTO tools (name) VALUES (?)", (tool,))
            except:
                pass
        conn.commit()
        conn.close()
    except Exception:
        pass

def is_allowed_thumb(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_THUMB_EXTS

def save_thumbnail(file_storage) -> str:
    """
    Saves and optimizes thumbnail.
    Reliably converts GIFs to Animated WebP.
    """
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
                for i in range(60):  # Limit to 60 frames
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
            if img.mode != 'RGB':
                img = img.convert('RGB')
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
    if not existing_rel_path:
        return None
    src = BASE_DIR / "static" / existing_rel_path
    if not src.exists():
        return None
    ext = src.suffix.lower()
    new_name = f"{uuid.uuid4().hex}{ext}"
    dest = UPLOAD_DIR / new_name
    try:
        shutil.copyfile(src, dest)
        return f"uploads/thumbs/{new_name}"
    except:
        return None

def delete_thumbnail_if_unused(rel_path: str) -> None:
    if not rel_path:
        return
    conn = get_db()
    count = conn.execute("SELECT COUNT(*) FROM prompts WHERE thumbnail = ?", (rel_path,)).fetchone()[0]
    conn.close()
    if count <= 1:
        try:
            (BASE_DIR / "static" / rel_path).unlink()
        except:
            pass

def find_placeholders(text: str) -> list[str]:
    matches = re.findall(r"\[\[(.+?)\]\]", text)
    return list(dict.fromkeys(matches))


# -------------------------
# Use Prompt rendering helpers
# -------------------------

_PLACEHOLDER_RE = re.compile(r"\[\[(.+?)\]\]")

def normalise_placeholder_name(name: str) -> str:
    """Normalise placeholder name for stable mapping.

    - Strip outer whitespace
    - Collapse internal whitespace to single spaces
    - Casefold to handle casing differences
    """
    if name is None:
        return ""
    s = str(name)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()

def extract_placeholders_meta(text: str) -> list[dict]:
    """Return unique placeholders with display + normalised key, preserving order."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[dict] = []
    for m in _PLACEHOLDER_RE.finditer(text):
        display = m.group(1)
        key = normalise_placeholder_name(display)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({
            "display": display,
            "key": key,
        })
    return out

def cleanup_placeholder_whitespace(s: str) -> str:
    """Tidy spaces introduced by placeholder removals while preserving newlines."""
    if s is None:
        return ""
    # Trim trailing spaces per line
    lines = [ln.rstrip() for ln in str(s).splitlines()]
    s = "\n".join(lines)

    # Collapse repeated spaces (not newlines)
    s = re.sub(r"[ \t]{2,}", " ", s)

    # Remove space before punctuation
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)

    # Clean up spacing around newlines
    s = re.sub(r" *\n *", "\n", s)

    # Avoid leading/trailing whitespace
    return s.strip()

def render_use_output(base_content: str, fills: dict[str, str], final_override: str | None = None) -> str:
    """Render the final prompt text for Use Prompt.

    Order:
    1) Expand descriptor tokens ({{descriptor:id}}) to rendered_text
    2) Replace placeholders [[...]] using normalised placeholder key lookup
    3) Remove any remaining placeholders (unfilled) by replacing them with ""
    4) Clean up whitespace
    """
    if final_override is not None and str(final_override).strip():
        working = str(final_override)
    else:
        working = str(base_content or "")

    # 1) Token expansion (server-side source of truth)
    try:
        working = inject_descriptors(working)
    except Exception:
        # Keep output usable even if token expansion fails for some reason
        pass

    # 2 + 3) Placeholders
    def _sub(m: re.Match) -> str:
        raw = m.group(1)
        key = normalise_placeholder_name(raw)
        val = (fills.get(key) or "").strip()
        return val  # empty string removes unfilled placeholders

    working = _PLACEHOLDER_RE.sub(_sub, working)

    # 4) Clean up
    return cleanup_placeholder_whitespace(working)

def parse_use_fills_from_form(form, placeholders_meta: list[dict]) -> dict[str, str]:
    """Parse placeholder fill selections from the Use Prompt POST payload."""
    fills: dict[str, str] = {}
    for i, ph in enumerate(placeholders_meta):
        key = ph.get("key", "")
        if not key:
            continue

        mode = (form.get(f"ph_mode_{i}") or "text").strip().lower()
        if mode == "descriptor":
            val = (form.get(f"ph_desc_text_{i}") or "").strip()
        else:
            val = (form.get(f"ph_text_{i}") or "").strip()

        if val:
            fills[key] = val
        else:
            # Explicitly set to empty if present, so unfilled placeholders are removed consistently
            fills.setdefault(key, "")
    return fills

def parse_prompt_txt(raw_text: str) -> dict:
    meta = {}
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
    return {
        "title": meta.get("title", ""),
        "category": meta.get("category", ""),
        "tool": meta.get("tool", ""),
        "prompt_type": meta.get("type", ""),
        "notes": meta.get("notes", ""),
        "content": "\n".join(lines[content_start:]).strip(),
    }


# -------------------------
# EXPORT / IMPORT LOGIC
# -------------------------

# ----------------------------
# Descriptors (generic entities)
# ----------------------------

def _parse_json_maybe(value: str, default):
    try:
        return json.loads(value) if value else default
    except Exception:
        return default

def get_descriptor_packs(pack_type: str | None = None):
    with get_db() as conn:
        if pack_type:
            return conn.execute(
                "SELECT * FROM descriptor_packs WHERE pack_type = ? ORDER BY name COLLATE NOCASE",
                (pack_type,),
            ).fetchall()
        return conn.execute("SELECT * FROM descriptor_packs ORDER BY pack_type, name COLLATE NOCASE").fetchall()

def get_descriptor_pack(pack_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM descriptor_packs WHERE id = ?", (pack_id,)).fetchone()

def get_descriptors(pack_type: str | None = None, q: str | None = None):
    q_like = f"%{q.strip()}%" if q else None
    with get_db() as conn:
        if pack_type and q_like:
            return conn.execute(
                """
                SELECT d.*, p.name AS pack_name
                FROM descriptors d
                JOIN descriptor_packs p ON p.id = d.pack_id
                WHERE d.pack_type = ?
                  AND (d.title LIKE ? OR d.rendered_text LIKE ? OR IFNULL(d.notes,'') LIKE ?)
                ORDER BY d.updated_at DESC
                """,
                (pack_type, q_like, q_like, q_like),
            ).fetchall()
        if pack_type:
            return conn.execute(
                """
                SELECT d.*, p.name AS pack_name
                FROM descriptors d
                JOIN descriptor_packs p ON p.id = d.pack_id
                WHERE d.pack_type = ?
                ORDER BY d.updated_at DESC
                """,
                (pack_type,),
            ).fetchall()
        if q_like:
            return conn.execute(
                """
                SELECT d.*, p.name AS pack_name
                FROM descriptors d
                JOIN descriptor_packs p ON p.id = d.pack_id
                WHERE (d.title LIKE ? OR d.rendered_text LIKE ? OR IFNULL(d.notes,'') LIKE ?)
                ORDER BY d.updated_at DESC
                """,
                (q_like, q_like, q_like),
            ).fetchall()

        return conn.execute(
            """
            SELECT d.*, p.name AS pack_name
            FROM descriptors d
            JOIN descriptor_packs p ON p.id = d.pack_id
            ORDER BY d.updated_at DESC
            """
        ).fetchall()

def get_descriptor(descriptor_id: int):
    with get_db() as conn:
        return conn.execute("SELECT * FROM descriptors WHERE id = ?", (descriptor_id,)).fetchone()

def normalise_pack(pack: dict) -> dict:
    """
    Ensure pack has predictable structure.
    Expected keys: fields_order (list), fields (dict), templates (dict), default_template (optional).
    """
    if not isinstance(pack, dict):
        return {"fields_order": [], "fields": {}, "templates": {"default": ""}, "default_template": "default"}

    fields = pack.get("fields") or {}
    templates = pack.get("templates") or {}
    fields_order = pack.get("fields_order") or list(fields.keys())
    default_template = pack.get("default_template") or ("default" if "default" in templates else (next(iter(templates.keys()), "default")))

    # coerce
    if not isinstance(fields, dict):
        fields = {}
    if not isinstance(templates, dict):
        templates = {"default": ""}
    if not isinstance(fields_order, list):
        fields_order = list(fields.keys())

    # guarantee every field_order key exists in fields
    for k in list(fields_order):
        if k not in fields:
            fields[k] = {"label": k, "options": []}

    # ensure at least one template
    if not templates:
        templates = {"default": ""}

    if default_template not in templates:
        default_template = next(iter(templates.keys()))

    pack["fields"] = fields
    pack["fields_order"] = fields_order
    pack["templates"] = templates
    pack["default_template"] = default_template
    return pack

def render_from_pack(pack: dict, values: dict, template_key: str | None = None) -> str:
    pack = normalise_pack(pack)
    templates = pack.get("templates", {})
    key = template_key or pack.get("default_template") or "default"
    template = templates.get(key) or templates.get("default") or ""
    # Safe formatting: missing keys become empty string
    class SafeDict(dict):
        def __missing__(self, k):
            return ""
    return template.format_map(SafeDict(values or {})).strip()



# -------------------------
# Thumbnail Serving (compat)
# -------------------------
# Some templates expect url_for('serve_thumbnail', prompt_id=...)
# This endpoint serves the prompt's thumbnail file if present, otherwise a 1x1 transparent PNG.
_TRANSPARENT_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/w8AAn8B9p2LhQAAAABJRU5ErkJggg=="
)

@app.route("/thumb/<int:prompt_id>")
def serve_thumbnail(prompt_id: int):
    try:
        db = get_db()
        row = db.execute("SELECT thumbnail FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        rel = row["thumbnail"] if row else None

        if rel:
            # common: 'uploads/thumbs/xyz.webp' stored relative to /static
            p = BASE_DIR / "static" / rel
            if p.exists():
                return send_file(p)

            # allow accidental leading 'static/'
            rel2 = rel.replace("static/", "", 1)
            p2 = BASE_DIR / "static" / rel2
            if p2.exists():
                return send_file(p2)

        png_bytes = base64.b64decode(_TRANSPARENT_PNG_B64)
        return send_file(io.BytesIO(png_bytes), mimetype="image/png")
    except Exception:
        png_bytes = base64.b64decode(_TRANSPARENT_PNG_B64)
        return send_file(io.BytesIO(png_bytes), mimetype="image/png")


@app.route("/export/selected", methods=["POST"])
def export_selected():
    ids = request.form.getlist("prompt_ids")
    if not ids:
        flash("No prompts selected for export.")
        return redirect(url_for("index"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"export_{ts}.db"
    export_db_path = TEMP_DIR / export_filename

    conn_exp = sqlite3.connect(export_db_path)
    conn_exp.execute("CREATE TABLE IF NOT EXISTS prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, category TEXT, tool TEXT, prompt_type TEXT, content TEXT, notes TEXT, thumbnail TEXT, parent_id INTEGER, group_id INTEGER, created_at TEXT, updated_at TEXT)")
    conn_exp.execute("CREATE TABLE IF NOT EXISTS tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE)")
    conn_exp.execute("CREATE TABLE IF NOT EXISTS prompt_tags (prompt_id INTEGER, tag_id INTEGER)")

    conn_main = get_db()
    placeholders = ",".join("?" * len(ids))
    prompts = conn_main.execute(f"SELECT * FROM prompts WHERE id IN ({placeholders})", ids).fetchall()

    for p in prompts:
        p_data = dict(p)
        cols = ",".join(p_data.keys())
        qmarks = ",".join("?" * len(p_data))
        conn_exp.execute(f"INSERT INTO prompts ({cols}) VALUES ({qmarks})", list(p_data.values()))

        pt_rows = conn_main.execute("SELECT t.name FROM tags t JOIN prompt_tags pt ON t.id = pt.tag_id WHERE pt.prompt_id=?", (p["id"],)).fetchall()

        for tag_row in pt_rows:
            t_name = tag_row["name"]
            try:
                conn_exp.execute("INSERT INTO tags (name) VALUES (?)", (t_name,))
            except sqlite3.IntegrityError:
                pass

            t_id_exp = conn_exp.execute("SELECT id FROM tags WHERE name=?", (t_name,)).fetchone()[0]
            conn_exp.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (p["id"], t_id_exp))

    conn_exp.commit()
    conn_exp.close()
    conn_main.close()

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(export_db_path, arcname="prompts.export.db")
        for p in prompts:
            if p["thumbnail"]:
                thumb_path = BASE_DIR / "static" / p["thumbnail"]
                if thumb_path.exists():
                    zf.write(thumb_path, arcname=p["thumbnail"])

    try:
        os.remove(export_db_path)
    except:
        pass

    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype="application/zip", as_attachment=True, download_name=f"prompts_export_{ts}.zip")


@app.route("/import/db", methods=["GET", "POST"])
def import_db_page():
    if request.method == "GET":
        return render_template("import_db.html", stage="upload")

    f = request.files.get("import_file")
    if not f:
        return redirect(url_for("import_db_page"))

    filename = secure_filename(f.filename)
    temp_path = TEMP_DIR / f"import_{uuid.uuid4().hex}_{filename}"
    f.save(temp_path)

    extracted_db_path = temp_path
    is_zip = filename.endswith(".zip")

    if is_zip:
        with zipfile.ZipFile(temp_path, 'r') as zf:
            db_files = [n for n in zf.namelist() if n.endswith(".db") or n.endswith(".sqlite")]
            if not db_files:
                flash("Invalid zip: No database file found.")
                return redirect(url_for("import_db_page"))

            extracted_db_path = TEMP_DIR / f"extracted_{uuid.uuid4().hex}.db"
            with open(extracted_db_path, "wb") as db_out:
                db_out.write(zf.read(db_files[0]))

    try:
        conn_imp = sqlite3.connect(extracted_db_path)
        conn_imp.row_factory = sqlite3.Row
        prompts = conn_imp.execute("SELECT * FROM prompts").fetchall()
        conn_imp.close()
    except Exception as e:
        flash(f"Error reading database: {e}")
        return redirect(url_for("import_db_page"))

    session["import_source_path"] = str(temp_path)
    session["import_extracted_db"] = str(extracted_db_path) if is_zip else str(temp_path)
    session["import_is_zip"] = is_zip

    return render_template("import_db.html", stage="preview", prompts=prompts)


@app.route("/import/db/commit", methods=["POST"])
def import_db_commit():
    selected_ids = request.form.getlist("prompt_ids")
    if not selected_ids:
        flash("No prompts selected.")
        return redirect(url_for("import_db_page"))

    source_zip_path = session.get("import_source_path")
    extracted_db_path = session.get("import_extracted_db")
    is_zip = session.get("import_is_zip")

    if not extracted_db_path or not os.path.exists(extracted_db_path):
        flash("Import session expired. Please start over.")
        return redirect(url_for("import_db_page"))

    conn_main = get_db()
    conn_imp = sqlite3.connect(extracted_db_path)
    conn_imp.row_factory = sqlite3.Row

    zip_ref = None
    if is_zip and source_zip_path:
        zip_ref = zipfile.ZipFile(source_zip_path, 'r')

    count = 0
    placeholders = ",".join("?" * len(selected_ids))
    prompts_to_import = conn_imp.execute(f"SELECT * FROM prompts WHERE id IN ({placeholders})", selected_ids).fetchall()

    for p in prompts_to_import:
        try:
            conn_main.execute("INSERT INTO categories (name) VALUES (?)", (p["category"],))
        except sqlite3.IntegrityError:
            pass

        try:
            conn_main.execute("INSERT INTO tools (name) VALUES (?)", (p["tool"],))
        except sqlite3.IntegrityError:
            pass

        final_thumb_path = None
        if p["thumbnail"] and is_zip and zip_ref:
            if p["thumbnail"] in zip_ref.namelist():
                img_data = zip_ref.read(p["thumbnail"])
                ext = Path(p["thumbnail"]).suffix
                new_filename = f"{uuid.uuid4().hex}{ext}"
                dest_path = UPLOAD_DIR / new_filename
                with open(dest_path, "wb") as f_out:
                    f_out.write(img_data)
                final_thumb_path = f"uploads/thumbs/{new_filename}"

        now = datetime.utcnow().isoformat()

        # Group mapping (optional)
        group_id_main = None
        try:
            if "group_id" in p.keys() and p["group_id"]:
                grp = conn_imp.execute("SELECT name, description FROM prompt_groups WHERE id=?", (p["group_id"],)).fetchone()
                if grp:
                    group_id_main = ensure_group_exists(grp["name"], grp["description"])
        except Exception:
            group_id_main = None

        cur = conn_main.execute(
            """INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, group_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (p["title"], p["category"], p["tool"], p["prompt_type"], p["content"], p["notes"], final_thumb_path, group_id_main, now, now)
        )
        new_prompt_id = cur.lastrowid

        pt_rows = conn_imp.execute("SELECT t.name FROM tags t JOIN prompt_tags pt ON t.id = pt.tag_id WHERE pt.prompt_id=?", (p["id"],)).fetchall()
        for t_row in pt_rows:
            tag_name = t_row["name"]
            try:
                conn_main.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
            except sqlite3.IntegrityError:
                pass

            tag_id_main = conn_main.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()[0]
            try:
                conn_main.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (new_prompt_id, tag_id_main))
            except sqlite3.IntegrityError:
                pass

        count += 1

    conn_main.commit()
    conn_main.close()
    conn_imp.close()
    if zip_ref:
        zip_ref.close()

    try:
        if source_zip_path:
            os.remove(source_zip_path)
        if extracted_db_path and extracted_db_path != source_zip_path:
            os.remove(extracted_db_path)
    except:
        pass

    session.pop("import_source_path", None)

    flash(f"Successfully imported {count} prompts.")
    return redirect(url_for("index"))


def build_library_base_query(args):
    """Builds FROM/JOIN/WHERE + params for the library query, matching index() filters."""
    category = args.get("category", "all")
    tool = args.get("tool", "all")
    group_id = args.get("group", "all")
    tag_filter = args.get("tag", "all")
    q = (args.get("q") or "").strip()
    q_not = (args.get("q_not") or "").strip()
    view_family = args.get("view_family")

    sort_by = args.get("sort_by", "updated_at")
    sort_dir = args.get("sort_dir", "desc")

    allowed_sorts = {"updated_at", "created_at", "title", "category", "tool", "prompt_type"}
    if sort_by not in allowed_sorts:
        sort_by = "updated_at"
    if sort_dir not in {"asc", "desc"}:
        sort_dir = "desc"

    base_query = " FROM prompts p"
    params = []

    if tag_filter != "all":
        base_query += " JOIN prompt_tags pt ON p.id = pt.prompt_id JOIN tags t ON pt.tag_id = t.id"

    base_query += " WHERE 1=1"

    if view_family:
        base_query += " AND (p.id = ? OR p.parent_id = ?)"
        params.extend([view_family, view_family])
    elif not q and not q_not:
        base_query += " AND p.parent_id IS NULL"

    if category != "all":
        base_query += " AND p.category = ?"
        params.append(category)
    if tool != "all":
        base_query += " AND p.tool = ?"
        params.append(tool)
    if group_id != "all":
        base_query += " AND p.group_id = ?"
        params.append(group_id)
    if tag_filter != "all":
        base_query += " AND t.name = ?"
        params.append(tag_filter)

    if q:
        terms = [t.strip() for t in q.split(",") if t.strip()]
        for term in terms:
            sub = "(LOWER(p.title) LIKE ? OR LOWER(p.content) LIKE ? OR LOWER(p.notes) LIKE ? OR EXISTS (SELECT 1 FROM prompt_tags pt2 JOIN tags t2 ON t2.id = pt2.tag_id WHERE pt2.prompt_id=p.id AND LOWER(t2.name) LIKE ?))"
            base_query += f" AND {sub}"
            lt = f"%{term.lower()}%"
            params.extend([lt, lt, lt, lt])

    if q_not:
        terms = [t.strip() for t in q_not.split(",") if t.strip()]
        for term in terms:
            sub = "(LOWER(p.title) NOT LIKE ? AND LOWER(p.content) NOT LIKE ? AND LOWER(p.notes) NOT LIKE ? AND NOT EXISTS (SELECT 1 FROM prompt_tags pt2 JOIN tags t2 ON t2.id = pt2.tag_id WHERE pt2.prompt_id=p.id AND LOWER(t2.name) LIKE ?))"
            base_query += f" AND {sub}"
            lt = f"%{term.lower()}%"
            params.extend([lt, lt, lt, lt])

    # --- NSFW visibility filter (default hidden) ---
    base_query, params = apply_nsfw_filter(base_query, params)

    return {
        "base_query": base_query,
        "params": params,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "category": category,
        "tool": tool,
        "group_id": group_id,
        "tag_filter": tag_filter,
        "q": q,
        "q_not": q_not,
        "view_family": view_family,
    }


# -------------------------
# NSFW toggle routes (session-level)
# -------------------------
@app.route("/nsfw/unlock", methods=["POST"])
def nsfw_unlock():
    session["show_nsfw"] = True
    return redirect(request.referrer or url_for("index"))

@app.route("/nsfw/lock", methods=["POST"])
def nsfw_lock():
    session["show_nsfw"] = False
    return redirect(request.referrer or url_for("index"))


# -------------------------
# Routes (Standard)
# -------------------------
@app.route("/", methods=["GET"])
def index():
    category = request.args.get("category", "all")
    tool = request.args.get("tool", "all")
    group_id = request.args.get("group", "all")
    tag_filter = request.args.get("tag", "all")
    q = (request.args.get("q") or "").strip()
    q_not = (request.args.get("q_not") or "").strip()
    view_family = request.args.get("view_family")  # NEW: ID to drill down into

    sort_by = request.args.get("sort_by", "updated_at")
    sort_dir = request.args.get("sort_dir", "desc")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except:
        page = 1

    ALLOWED_SORTS = {"updated_at", "created_at", "title", "category", "tool", "prompt_type"}
    if sort_by not in ALLOWED_SORTS:
        sort_by = "updated_at"
    if sort_dir not in ["asc", "desc"]:
        sort_dir = "desc"

    conn = get_db()

    base_query = " FROM prompts p"
    if tag_filter != "all":
        base_query += " JOIN prompt_tags pt ON p.id = pt.prompt_id JOIN tags t ON pt.tag_id = t.id"

    base_query += " WHERE 1=1"
    params = []

    # --- NEW: VISIBILITY LOGIC ---
    if view_family:
        base_query += " AND (p.id = ? OR p.parent_id = ?)"
        params.extend([view_family, view_family])
    elif not q and not q_not and tag_filter == "all":
        base_query += " AND p.parent_id IS NULL"
    # -----------------------------

    if category != "all":
        base_query += " AND p.category = ?"
        params.append(category)
    if tool != "all":
        base_query += " AND p.tool = ?"
        params.append(tool)
    if group_id != "all":
        base_query += " AND p.group_id = ?"
        params.append(group_id)
    if tag_filter != "all":
        base_query += " AND t.name = ?"
        params.append(tag_filter)

    if q:
        terms = [t.strip() for t in q.split(',') if t.strip()]
        for term in terms:
            sub = "(LOWER(p.title) LIKE ? OR LOWER(p.content) LIKE ? OR LOWER(COALESCE(p.notes, '')) LIKE ? OR EXISTS (SELECT 1 FROM prompt_tags pt JOIN tags t2 ON pt.tag_id=t2.id WHERE pt.prompt_id=p.id AND LOWER(t2.name) LIKE ?))"
            base_query += f" AND {sub}"
            lt = f"%{term.lower()}%"
            params.extend([lt, lt, lt, lt])

    if q_not:
        terms = [t.strip() for t in q_not.split(',') if t.strip()]
        for term in terms:
            sub = "NOT (LOWER(p.title) LIKE ? OR LOWER(p.content) LIKE ? OR LOWER(COALESCE(p.notes, '')) LIKE ? OR EXISTS (SELECT 1 FROM prompt_tags pt JOIN tags t2 ON pt.tag_id=t2.id WHERE pt.prompt_id=p.id AND LOWER(t2.name) LIKE ?))"
            base_query += f" AND {sub}"
            lt = f"%{term.lower()}%"
            params.extend([lt, lt, lt, lt])

    # --- NSFW visibility filter (default hidden) ---
    base_query, params = apply_nsfw_filter(base_query, params)

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

    prompts = [dict(r) for r in rows]
    ids = [p["id"] for p in prompts]
    tag_map = get_tags_for_prompts(conn, ids)
    group_name_map = get_group_name_map(conn, [p.get("group_id") for p in prompts])
    for p in prompts:
        p["tags"] = tag_map.get(p["id"], [])
        p["group_name"] = group_name_map.get(p.get("group_id"))
        # Optional: expand descriptor tokens for display/preview
        p["content"] = inject_descriptors(p.get("content") or "")

        # Optional: expand descriptor tokens for display/preview
        p["content"] = inject_descriptors(p.get("content") or "")


    saved_views = get_saved_views()
    all_tags = get_all_tags()
    groups = get_groups()
    conn.close()

    return render_template(
        "index.html",
        prompts=prompts,
        categories=get_categories(),
        tools=get_tools(),
        groups=groups,
        active_category=category,
        active_tool=tool,
        active_group=group_id,
        active_tag=tag_filter,
        q=q,
        q_not=q_not,
        view_family=view_family,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        total_pages=total_pages,
        total_items=total_items,
        saved_views=saved_views,
        all_tags=all_tags
    )


@app.route("/api/prompts", methods=["GET"])
def api_prompts():
    try:
        page = max(1, int(request.args.get("page", 1)))
    except Exception:
        page = 1

    try:
        per_page = max(1, min(60, int(request.args.get("per_page", ITEMS_PER_PAGE))))
    except Exception:
        per_page = ITEMS_PER_PAGE

    conn = get_db()
    qinfo = build_library_base_query(request.args)
    base_query = qinfo["base_query"]
    params = qinfo["params"]
    sort_by = qinfo["sort_by"]
    sort_dir = qinfo["sort_dir"]

    offset = (page - 1) * per_page

    data_query = f"""
        SELECT DISTINCT p.*,
        (SELECT COUNT(*) FROM prompts AS p2 WHERE p2.parent_id = p.id) as child_count
        {base_query}
        ORDER BY p.{sort_by} {sort_dir.upper()}
        LIMIT ? OFFSET ?
    """

    rows = conn.execute(data_query, params + [per_page + 1, offset]).fetchall()

    has_more = len(rows) > per_page
    rows = rows[:per_page]

    prompts = [dict(r) for r in rows]

    ids = [p["id"] for p in prompts]
    tag_map = get_tags_for_prompts(conn, ids)
    group_name_map = get_group_name_map(conn, [p.get("group_id") for p in prompts])
    for p in prompts:
        p["tags"] = tag_map.get(p["id"], [])
        p["group_name"] = group_name_map.get(p.get("group_id"))

    conn.close()

    html = render_template("_prompt_cards.html", prompts=prompts)

    return jsonify({
        "html": html,
        "has_more": has_more,
        "next_page": (page + 1) if has_more else None
    })

@app.route("/api/prompt/<int:prompt_id>/upload_thumb", methods=["POST"])
def upload_thumb_api(prompt_id):
    f = request.files.get("file")
    if not f or not is_allowed_thumb(f.filename):
        return jsonify({"error": "Invalid file type"}), 400

    try:
        new_path = save_thumbnail(f)

        now = datetime.utcnow().isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE prompts SET thumbnail = ?, updated_at = ? WHERE id = ?",
                (new_path, now, prompt_id)
            )
            conn.commit()

        return jsonify({
            "success": True,
            "thumbnail_url": url_for('static', filename=new_path),
            "thumbnail_path": new_path
        })

    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/view/save", methods=["POST"])
def save_view():
    name = request.form.get("view_name", "").strip()
    query_string = request.form.get("query_string", "")
    if not name or not query_string:
        return redirect(url_for("index"))

    conn = get_db()
    conn.execute("INSERT INTO saved_views (name, query_params, created_at) VALUES (?, ?, ?)",
                 (name, query_string, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    flash(f"View '{name}' saved.")
    return redirect(f"/?{query_string}")

@app.route("/view/delete/<int:view_id>", methods=["POST"])
def delete_view(view_id):
    conn = get_db()
    conn.execute("DELETE FROM saved_views WHERE id=?", (view_id,))
    conn.commit()
    conn.close()
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

        # Optional theme group
        group_id = request.form.get("group_id", "")
        new_group_name = request.form.get("new_group_name", "").strip()
        if new_group_name:
            group_id = ensure_group_exists(new_group_name)
        group_id = int(group_id) if str(group_id).isdigit() else None

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
            INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, group_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, category, tool, prompt_type, content, notes, thumb_path, group_id, now, now),
        )
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        save_tags(conn, new_id, tags_str)
        conn.commit()
        conn.close()
        return redirect(url_for("index"))

    return render_template(
        "edit_prompt.html",
        prompt=None,
        categories=categories,
        tools=tools,
        groups=get_groups(),
        prompt_types=PROMPT_TYPES,
        tags_str="",
        ollama_models=ollama_models
    )


@app.route("/prompt/<int:prompt_id>/edit", methods=["GET", "POST"])
def edit_prompt(prompt_id: int):
    prompt = get_prompt(prompt_id)
    if not prompt:
        flash("Prompt not found.")
        return redirect(url_for("index"))

    if request.method == "POST":
        title = request.form.get("title", "").strip()
        category = request.form.get("category", "Other").strip()
        tool = request.form.get("tool", "Generic").strip()
        ensure_category_tool_exist(category, tool)

        prompt_type = request.form.get("prompt_type", "Instruction").strip()
        content = request.form.get("content", "").strip()
        notes = request.form.get("notes", "").strip()
        tags_str = request.form.get("tags", "").strip()

        group_id = request.form.get("group_id", "")
        new_group_name = request.form.get("new_group_name", "").strip()
        if new_group_name:
            group_id = ensure_group_exists(new_group_name)
        group_id = int(group_id) if str(group_id).isdigit() else None

        new_thumb = prompt["thumbnail"]

        if request.form.get("remove_thumbnail") == "on":
            delete_thumbnail_if_unused(new_thumb)
            new_thumb = None

        hidden_thumb = request.form.get("hidden_thumbnail_path")
        if hidden_thumb:
            if hidden_thumb != prompt["thumbnail"]:
                if new_thumb:
                    delete_thumbnail_if_unused(new_thumb)
                new_thumb = hidden_thumb

        f = request.files.get("thumbnail")
        if f and f.filename and is_allowed_thumb(f.filename):
            if new_thumb:
                delete_thumbnail_if_unused(new_thumb)
            new_thumb = save_thumbnail(f)

        now = datetime.utcnow().isoformat()

        conn = get_db()
        conn.execute(
            """
            UPDATE prompts
            SET title=?, category=?, tool=?, prompt_type=?, content=?, notes=?, thumbnail=?, group_id=?, updated_at=?
            WHERE id=?
            """,
            (title, category, tool, prompt_type, content, notes, new_thumb, group_id, now, prompt_id),
        )
        save_tags(conn, prompt_id, tags_str)
        conn.commit()
        conn.close()

        return redirect(url_for("index"))

    categories = get_categories()
    tools = get_tools()
    ollama_models = ollama_list_models()

    conn = get_db()
    current_tags = get_tags_for_prompt(conn, prompt_id)
    conn.close()

    return render_template(
        "edit_prompt.html",
        prompt=prompt,
        categories=categories,
        tools=tools,
        groups=get_groups(),
        prompt_types=PROMPT_TYPES,
        tags_str=", ".join(current_tags),
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

    submitted_title = request.form.get("new_title") or request.form.get("title")
    if submitted_title and as_variant:
        new_title = submitted_title
    else:
        new_title = f"{p['title']} (Variant)" if as_variant else f"Copy of {p['title']}"

    parent_id = p["parent_id"] if (as_variant and p["parent_id"]) else (p["id"] if as_variant else None)

    new_thumb = None
    if p["thumbnail"]:
        new_thumb = copy_thumbnail(p["thumbnail"])

    new_content = request.form.get("content")
    if new_content is None:
        new_content = p["content"]

    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, group_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (new_title, p["category"], p["tool"], p["prompt_type"], new_content, p["notes"], new_thumb, parent_id, p["group_id"], now, now),
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

    if not prompt:
        conn.close()
        flash("Prompt not found.")
        return redirect(url_for("index"))

    # --- NSFW direct access protection ---
    if NSFW_LOCK_ENABLED_DEFAULT and not nsfw_is_unlocked():
        if prompt_is_nsfw(conn, prompt_id):
            conn.close()
            flash("This prompt is locked (NSFW). Unlock NSFW visibility to view it.")
            return redirect(url_for("index"))

    root_id = prompt["parent_id"] if prompt["parent_id"] else prompt["id"]

    variants = conn.execute("""
        SELECT id, title, prompt_type
        FROM prompts
        WHERE id = ? OR parent_id = ?
        ORDER BY id ASC
    """, (root_id, root_id)).fetchall()

    conn.close()

    content = prompt["content"]
    base_text = inject_descriptors(content)
    placeholders_meta = extract_placeholders_meta(content)
    final_text = None

    if request.method == "POST":
        fills = parse_use_fills_from_form(request.form, placeholders_meta)

        override_active = (request.form.get("override_active") or "").strip() == "1"
        override_text = (request.form.get("final_override") or "").strip() if override_active else None

        final_text = render_use_output(content, fills, final_override=override_text)

    return render_template(
        "render_prompt.html",
        prompt=prompt,
        variants=variants,
        placeholders_meta=placeholders_meta,
        base_text=base_text,
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

        def safe_add_meta(c, t, active_conn):
            if c:
                try:
                    active_conn.execute("INSERT INTO categories (name) VALUES (?)", (c,))
                except sqlite3.IntegrityError:
                    pass
            if t:
                try:
                    active_conn.execute("INSERT INTO tools (name) VALUES (?)", (t,))
                except sqlite3.IntegrityError:
                    pass

        batch_type = request.form.get("batch_prompt_type", "Instruction").strip()
        batch_notes = request.form.get("batch_notes", "").strip()
        single_mode = request.form.get("single_prompt_per_file") == "on"
        use_first_line_as_title = request.form.get("use_first_line_as_title") == "on"
        prefer_metadata = request.form.get("prefer_file_metadata") == "on"

        conn = get_db()
        safe_add_meta(batch_cat, batch_tool, conn)

        now = datetime.utcnow().isoformat()

        for f in files:
            if not f or not f.filename:
                continue
            filename = secure_filename(f.filename)
            if not filename.lower().endswith(".txt"):
                continue

            raw = f.read().decode("utf-8", errors="ignore")
            chunks = [raw] if single_mode else re.split(r'\n\s*\n', raw)

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
                p = parse_prompt_txt(chunk)

                title = filename.rsplit(".", 1)[0]
                if p["title"]:
                    title = p["title"]
                elif use_first_line_as_title:
                    lines = p["content"].split('\n', 1)
                    if lines[0].strip():
                        title = lines[0].strip()
                        p["content"] = lines[1].strip() if len(lines) > 1 else ""

                cat = "Image"
                if prefer_metadata and p["category"]:
                    cat = p["category"]
                elif batch_cat:
                    cat = batch_cat
                elif p["category"]:
                    cat = p["category"]

                tool = "Generic"
                if prefer_metadata and p["tool"]:
                    tool = p["tool"]
                elif batch_tool:
                    tool = batch_tool
                elif p["tool"]:
                    tool = p["tool"]

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
    if not f:
        return redirect(url_for("import_prompt"))

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
    return render_template(
        "manage.html",
        categories=get_categories(),
        tools=get_tools(),
        groups=get_groups(),
    )

@app.route("/manage/group/add", methods=["POST"])
def add_group():
    name = request.form.get("name", "").strip()
    description = request.form.get("description", "").strip() or None
    if name:
        ensure_group_exists(name, description)
    return redirect(url_for("manage"))

@app.route("/manage/group/delete", methods=["POST"])
def delete_group():
    gid = request.form.get("id", "").strip()
    if gid.isdigit():
        with get_db() as conn:
            conn.execute("UPDATE prompts SET group_id = NULL WHERE group_id = ?", (int(gid),))
            conn.execute("DELETE FROM prompt_groups WHERE id = ?", (int(gid),))
            conn.commit()
    return redirect(url_for("manage"))

@app.route("/manage/category/add", methods=["POST"])
def add_category():
    n = request.form.get("name", "").strip()
    if n:
        ensure_category_tool_exist(n, None)
    return redirect(url_for("manage"))

@app.route("/manage/category/delete", methods=["POST"])
def delete_category():
    n = request.form.get("name", "").strip()
    if n:
        with get_db() as conn:
            conn.execute("DELETE FROM categories WHERE name=?", (n,))
            conn.commit()
    return redirect(url_for("manage"))

@app.route("/manage/tool/add", methods=["POST"])
def add_tool():
    n = request.form.get("name", "").strip()
    if n:
        ensure_category_tool_exist(None, n)
    return redirect(url_for("manage"))

@app.route("/manage/tool/delete", methods=["POST"])
def delete_tool():
    n = request.form.get("name", "").strip()
    if n:
        with get_db() as conn:
            conn.execute("DELETE FROM tools WHERE name=?", (n,))
            conn.commit()
    return redirect(url_for("manage"))


# ----------------------------
# Descriptor Packs UI
# ----------------------------

@app.route("/descriptor_packs", methods=["GET", "POST"])
def descriptor_packs():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip() or "Unnamed Pack"
        pack_type = (request.form.get("pack_type") or "generic").strip() or "generic"

        pack_json_text = (request.form.get("pack_json") or "").strip()

        # Optional file upload
        if not pack_json_text and "pack_file" in request.files:
            f = request.files["pack_file"]
            if f and f.filename:
                pack_json_text = f.read().decode("utf-8", errors="ignore").strip()

        try:
            raw_obj = json.loads(pack_json_text) if pack_json_text else {}
            bundle_descriptors = None

            # Support a "bundle" import format:
            # { "pack": {...meta...}, "descriptors": [ {...}, ... ] }
            if isinstance(raw_obj, dict) and ("pack" in raw_obj) and ("descriptors" in raw_obj):
                pack_meta = raw_obj.get("pack") or {}
                bundle_descriptors = raw_obj.get("descriptors") or []

                # Allow the bundle to override the form fields
                name = (pack_meta.get("name") or pack_meta.get("title") or name).strip() or name
                pack_type = (pack_meta.get("pack_type") or pack_meta.get("type") or pack_type).strip() or pack_type

                pack_obj = {
                    "schema_version": 1,
                    "pack_type": pack_type,
                    "meta": {
                        "name": name,
                        "description": pack_meta.get("description") or "",
                    },
                    "fields_order": ["text"],
                    "fields": {
                        "text": {
                            "label": "Text",
                            "type": "textarea",
                            "required": True,
                            "placeholder": "Write the descriptor text here...",
                        }
                    },
                    "templates": {
                        "default": "{text}"
                    },
                    "default_template": "default",
                }
            else:
                pack_obj = raw_obj

                # If the imported JSON contains a pack_type/type, prefer it over the form value
                if isinstance(pack_obj, dict):
                    inferred_type = (
                        (pack_obj.get("pack_type") or pack_obj.get("type"))
                        or ((pack_obj.get("meta") or {}).get("pack_type"))
                        or ((pack_obj.get("meta") or {}).get("type"))
                    )
                    if inferred_type:
                        pack_type = str(inferred_type).strip() or pack_type

            pack_obj = normalise_pack(pack_obj)
            pack_json_text = json.dumps(pack_obj, indent=2, ensure_ascii=False)
        except Exception as e:
            flash(f"Invalid JSON pack: {e}", "danger")
            return redirect(url_for("descriptor_packs"))

        now = datetime.utcnow().isoformat(timespec="seconds")
        with get_db() as conn:
            cur = conn.execute(
                "INSERT INTO descriptor_packs (name, pack_type, pack_json, created_at, updated_at) VALUES (?,?,?,?,?)",
                (name, pack_type, pack_json_text, now, now),
            )
            pack_id = cur.lastrowid

            imported_count = 0
            if bundle_descriptors:
                for item in bundle_descriptors:
                    if not isinstance(item, dict):
                        continue
                    d_title = (item.get("title") or item.get("name") or "").strip() or f"{name} Descriptor"
                    # Accept a few possible fields from different generators
                    d_text = (
                        item.get("rendered_text")
                        or item.get("text")
                        or item.get("render")
                        or ""
                    )
                    if not d_text and isinstance(item.get("values"), dict):
                        d_text = item["values"].get("text") or ""

                    d_text = str(d_text).strip()
                    if not d_text:
                        # Skip empty entries
                        continue

                    values_json = json.dumps({"text": d_text}, ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO descriptors (title, pack_id, pack_type, values_json, rendered_text, notes, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                        (d_title, pack_id, pack_type, values_json, d_text, (item.get("notes") or ""), now, now),
                    )
                    imported_count += 1

            conn.commit()
        if 'imported_count' in locals() and imported_count:
            flash(f"Descriptor pack created and imported {imported_count} descriptors.", "success")
        else:
            flash("Descriptor pack created.", "success")
        return redirect(url_for("descriptor_packs"))

    return render_template(
        "descriptor_packs.html",
        packs=get_descriptor_packs(),
        pack_types=sorted({p["pack_type"] for p in get_descriptor_packs()} | {"generic", "character", "creature", "environment", "prop", "vehicle", "scene"}),
    )

@app.route("/descriptor_pack/<int:pack_id>/edit", methods=["GET", "POST"])
def edit_descriptor_pack(pack_id: int):
    pack_row = get_descriptor_pack(pack_id)
    if not pack_row:
        flash("Pack not found.", "danger")
        return redirect(url_for("descriptor_packs"))

    if request.method == "POST":
        name = (request.form.get("name") or "").strip() or pack_row["name"]
        pack_type = (request.form.get("pack_type") or pack_row["pack_type"] or "generic").strip() or "generic"
        pack_json_text = (request.form.get("pack_json") or "").strip()

        try:
            pack_obj = json.loads(pack_json_text) if pack_json_text else {}
            pack_obj = normalise_pack(pack_obj)
            pack_json_text = json.dumps(pack_obj, indent=2, ensure_ascii=False)
        except Exception as e:
            flash(f"Invalid JSON pack: {e}", "danger")
            return redirect(url_for("edit_descriptor_pack", pack_id=pack_id))

        now = datetime.utcnow().isoformat(timespec="seconds")
        with get_db() as conn:
            conn.execute(
                "UPDATE descriptor_packs SET name=?, pack_type=?, pack_json=?, updated_at=? WHERE id=?",
                (name, pack_type, pack_json_text, now, pack_id),
            )
            # Keep existing descriptors aligned with the new pack_type
            conn.execute(
                "UPDATE descriptors SET pack_type=?, updated_at=? WHERE pack_id=?",
                (pack_type, now, pack_id),
            )
            conn.commit()

        flash("Pack updated.", "success")
        return redirect(url_for("descriptor_packs"))

    return render_template(
        "edit_descriptor_pack.html",
        pack=pack_row,
    )

@app.route("/descriptor_pack/<int:pack_id>/delete", methods=["POST"])
def delete_descriptor_pack(pack_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM descriptor_packs WHERE id = ?", (pack_id,))
        conn.commit()
    flash("Pack deleted.", "success")
    return redirect(url_for("descriptor_packs"))

# ----------------------------
# Descriptors UI
# ----------------------------

@app.route("/descriptors", methods=["GET"])
def descriptors():
    pack_type = (request.args.get("type") or "").strip() or None
    q = (request.args.get("q") or "").strip() or None
    return render_template(
        "descriptors.html",
        descriptors=get_descriptors(pack_type=pack_type, q=q),
        packs=get_descriptor_packs(pack_type=pack_type) if pack_type else get_descriptor_packs(),
        pack_type=pack_type,
        q=q or "",
        pack_types=sorted({p["pack_type"] for p in get_descriptor_packs()} | {"generic", "character", "creature", "environment", "prop", "vehicle", "scene"}),
    )

@app.route("/descriptor/new", methods=["GET", "POST"])
def new_descriptor():
    pack_id = request.args.get("pack_id", type=int) or request.form.get("pack_id", type=int)
    pack_row = get_descriptor_pack(pack_id) if pack_id else None
    if not pack_row:
        flash("Choose a pack first.", "warning")
        return redirect(url_for("descriptors"))

    pack_obj = _parse_json_maybe(pack_row["pack_json"], {})
    pack_obj = normalise_pack(pack_obj)

    if request.method == "POST":
        title = (request.form.get("title") or "").strip() or "Untitled"
        notes = (request.form.get("notes") or "").strip()
        template_key = (request.form.get("template_key") or "").strip() or pack_obj.get("default_template")

        values = {}
        for field_key in pack_obj.get("fields_order", []):
            values[field_key] = (request.form.get(f"f_{field_key}") or "").strip()

        rendered = render_from_pack(pack_obj, values, template_key=template_key)
        if not rendered:
            flash("Rendered text is empty. Check your template and fields.", "warning")

        now = datetime.utcnow().isoformat(timespec="seconds")
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO descriptors (title, pack_id, pack_type, values_json, rendered_text, notes, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (title, pack_row["id"], pack_row["pack_type"], json.dumps(values, ensure_ascii=False), rendered, notes, now, now),
            )
            conn.commit()

        flash("Descriptor saved.", "success")
        return redirect(url_for("descriptors", type=pack_row["pack_type"]))

    return render_template(
        "edit_descriptor.html",
        mode="new",
        descriptor=None,
        pack=pack_row,
        pack_obj=pack_obj,
        values={},
        rendered_preview="",
        template_key=pack_obj.get("default_template", "default"),
    )

@app.route("/descriptor/<int:descriptor_id>/edit", methods=["GET", "POST"])
def edit_descriptor(descriptor_id: int):
    d = get_descriptor(descriptor_id)
    if not d:
        flash("Descriptor not found.", "danger")
        return redirect(url_for("descriptors"))

    pack_row = get_descriptor_pack(d["pack_id"])
    if not pack_row:
        flash("Pack not found for this descriptor.", "danger")
        return redirect(url_for("descriptors"))

    pack_obj = _parse_json_maybe(pack_row["pack_json"], {})
    pack_obj = normalise_pack(pack_obj)

    values = _parse_json_maybe(d["values_json"], {})
    template_key = pack_obj.get("default_template", "default")
    rendered_preview = d["rendered_text"]

    if request.method == "POST":
        title = (request.form.get("title") or "").strip() or d["title"]
        notes = (request.form.get("notes") or "").strip()
        template_key = (request.form.get("template_key") or "").strip() or template_key

        values = {}
        for field_key in pack_obj.get("fields_order", []):
            values[field_key] = (request.form.get(f"f_{field_key}") or "").strip()

        rendered_preview = render_from_pack(pack_obj, values, template_key=template_key)

        now = datetime.utcnow().isoformat(timespec="seconds")
        with get_db() as conn:
            conn.execute(
                """
                UPDATE descriptors
                SET title=?, values_json=?, rendered_text=?, notes=?, updated_at=?
                WHERE id=?
                """,
                (title, json.dumps(values, ensure_ascii=False), rendered_preview, notes, now, descriptor_id),
            )
            conn.commit()

        flash("Descriptor updated.", "success")
        return redirect(url_for("descriptors", type=d["pack_type"]))

    return render_template(
        "edit_descriptor.html",
        mode="edit",
        descriptor=d,
        pack=pack_row,
        pack_obj=pack_obj,
        values=values,
        rendered_preview=rendered_preview,
        template_key=template_key,
    )

@app.route("/descriptor/<int:descriptor_id>/delete", methods=["POST"])
def delete_descriptor(descriptor_id: int):
    d = get_descriptor(descriptor_id)
    with get_db() as conn:
        conn.execute("DELETE FROM descriptors WHERE id = ?", (descriptor_id,))
        conn.commit()
    flash("Descriptor deleted.", "success")
    if d:
        return redirect(url_for("descriptors", type=d["pack_type"]))
    return redirect(url_for("descriptors"))

# Quick server-side preview (optional)
@app.route("/api/descriptor/preview", methods=["POST"])
def api_descriptor_preview():
    data = request.get_json(silent=True) or {}
    pack_id = data.get("pack_id")
    values = data.get("values") or {}
    template_key = data.get("template_key")
    if not pack_id:
        return jsonify({"ok": False, "error": "pack_id required"}), 400

    pack_row = get_descriptor_pack(int(pack_id))
    if not pack_row:
        return jsonify({"ok": False, "error": "pack not found"}), 404

    pack_obj = _parse_json_maybe(pack_row["pack_json"], {})
    pack_obj = normalise_pack(pack_obj)
    rendered = render_from_pack(pack_obj, values, template_key=template_key)
    return jsonify({"ok": True, "rendered": rendered})




@app.route("/api/descriptors/list")
def api_descriptors_list():
    """Return a lightweight list of saved descriptors for the prompt editor picker."""
    # Normalise to lower-case because pack_type values in the DB are not guaranteed to be consistent.
    pack_type = (request.args.get("pack_type", "all") or "all").strip().lower()
    q = (request.args.get("q") or "").strip().lower()

    db = get_db()
    where = []
    params = []

    if pack_type and pack_type != "all":
        where.append("LOWER(pack_type) = ?")
        params.append(pack_type)

    if q:
        where.append("(LOWER(title) LIKE ? OR LOWER(rendered_text) LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])

    # Respect NSFW lock: hide obvious NSFW descriptors when locked.
    # Descriptors do not have a tags join table (like prompts), so use a conservative keyword filter.
    if not nsfw_is_unlocked():
        nsfw_clauses = []
        for t in NSFW_TAGS:
            t = (t or "").strip().lower()
            if not t:
                continue
            nsfw_clauses.append("(LOWER(title) LIKE ? OR LOWER(rendered_text) LIKE ?)")
            params.extend([f"%{t}%", f"%{t}%"])
        if nsfw_clauses:
            where.append("NOT (" + " OR ".join(nsfw_clauses) + ")")

    sql = "SELECT id, title, pack_type, rendered_text, updated_at FROM descriptors"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY updated_at DESC LIMIT 300"

    rows = db.execute(sql, params).fetchall()
    return jsonify([{
        "id": r["id"],
        "title": r["title"],
        "pack_type": r["pack_type"],
        "rendered_text": r["rendered_text"] or ""
    } for r in rows])


@app.route("/api/descriptors/types")
@app.route("/api/descriptors/types")
def api_descriptors_types():
    """Return all known descriptor pack types (for UI dropdowns)."""
    db = get_db()
    rows = db.execute(
        "SELECT DISTINCT pack_type FROM descriptor_packs WHERE pack_type IS NOT NULL AND pack_type != ''"
    ).fetchall()

    defaults = {"generic", "character", "creature", "environment", "scene", "prop", "vehicle", "other"}

    types = sorted(
        {(r["pack_type"] or "").strip() for r in rows if (r["pack_type"] or "").strip()}
        | defaults
    )
    return jsonify(types)



# ---------------------------
# Descriptor Remix
# ---------------------------

@app.route("/remix", methods=["GET"])
def descriptor_remix():
    # Hard gate: Remix is hidden/disabled unless Ollama is reachable and has at least one model.
    if not ollama_is_ready_cached():
        flash("Remix is unavailable because Ollama is not reachable or no models are loaded.")
        return redirect(url_for("index"))

    models = ollama_models_cached()
    return render_template("remix.html", models=models)


@app.route("/api/remix/randomise", methods=["POST"])
def api_remix_randomise():
    if not ollama_is_ready_cached():
        return jsonify({"ok": False, "error": "Ollama is not available."}), 400

    data = request.get_json(silent=True) or {}
    pack_type = (data.get("pack_type") or "all").strip()
    try:
        count = int(data.get("count") or 8)
    except Exception:
        count = 8
    count = max(1, min(count, 50))

    db = get_db()
    where = []
    params = []

    if pack_type and pack_type != "all":
        where.append("pack_type = ?")
        params.append(pack_type)

    if not nsfw_is_unlocked():
        # Same conservative keyword-based exclusion used by /api/descriptors/list
        nsfw_clauses = []
        for t in NSFW_TAGS:
            t = (t or "").strip().lower()
            if not t:
                continue
            nsfw_clauses.append("(LOWER(title) LIKE ? OR LOWER(rendered_text) LIKE ?)")
            params.extend([f"%{t}%", f"%{t}%"])
        if nsfw_clauses:
            where.append("NOT (" + " OR ".join(nsfw_clauses) + ")")

    sql = "SELECT id, title, pack_type, rendered_text FROM descriptors"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY RANDOM() LIMIT ?"
    params2 = params + [count]

    rows = db.execute(sql, params2).fetchall()
    items = [{
        "id": r["id"],
        "title": r["title"],
        "pack_type": r["pack_type"],
        "rendered_text": (r["rendered_text"] or "").strip()
    } for r in rows]

    raw = ", ".join([i["rendered_text"] for i in items if i["rendered_text"]])
    return jsonify({"ok": True, "items": items, "raw": raw})


@app.route("/api/remix/rewrite", methods=["POST"])
def api_remix_rewrite():
    if not ollama_is_ready_cached():
        return jsonify({"ok": False, "error": "Ollama is not available."}), 400

    data = request.get_json(silent=True) or {}
    raw = (data.get("raw") or "").strip()
    model = (data.get("model") or "").strip() or None
    user_system = (data.get("system_prompt") or "").strip()

    if not raw:
        return jsonify({"ok": False, "error": "Nothing to rewrite."}), 400

    # Default guidance aims to: deduplicate, smooth, and collapse fragments into a coherent prompt.
    default_system = (
        "You rewrite rough prompt fragments into a single clean, coherent prompt. "
        "Do NOT preserve the original sentence order if it reads awkwardly. "
        "Combine related details, remove repetition, and drop orphan fragments that don't fit. "
        "Keep the result concise but specific. "
        "Output only the rewritten prompt, no commentary."
    )

    # Non-negotiable constraints: never invent beyond the supplied descriptors.
    guardrail = (
        "NON-NEGOTIABLE RULES: "
        "Use ONLY attributes explicitly present in the input. "
        "Do NOT invent, infer, or add new attributes. "
        "If there are conflicts, resolve them by OMITTING conflicting details, not by inventing a compromise."
    )

    # Allow the user to steer style/format, but always enforce guardrails.
    # Limit system prompt length to avoid accidental mega-prompts.
    if user_system:
        user_system = user_system[:2000]
        system = f"{user_system}\n\n{guardrail}"
    else:
        system = f"{default_system}\n\n{guardrail}"

    try:
        rewritten = ollama_generate(raw, model=model, system=system)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Ollama error: {e}"}), 500

    return jsonify({"ok": True, "rewritten": rewritten})


@app.route("/bulk_update", methods=["POST"])
def bulk_update():
    ids = request.form.getlist("prompt_ids")
    cat = request.form.get("bulk_category")
    tool = request.form.get("bulk_tool")
    ptype = request.form.get("bulk_prompt_type")
    bulk_group = request.form.get("bulk_group")
    tags_str = request.form.get("bulk_tags", "").strip()

    ensure_category_tool_exist(cat, tool)

    if ids:
        with get_db() as conn:
            for pid in ids:
                now_str = datetime.utcnow().isoformat()
                if cat:
                    conn.execute("UPDATE prompts SET category=?, updated_at=? WHERE id=?", (cat, now_str, pid))
                if tool:
                    conn.execute("UPDATE prompts SET tool=?, updated_at=? WHERE id=?", (tool, now_str, pid))
                if ptype:
                    conn.execute("UPDATE prompts SET prompt_type=?, updated_at=? WHERE id=?", (ptype, now_str, pid))
                if bulk_group is not None and str(bulk_group).strip() != "":
                    bg = str(bulk_group).strip()
                    if bg.lower() == "none":
                        conn.execute("UPDATE prompts SET group_id=NULL, updated_at=? WHERE id=?", (now_str, pid))
                    else:
                        try:
                            gid = int(bg)
                        except ValueError:
                            gid = None
                        if gid is not None:
                            conn.execute("UPDATE prompts SET group_id=?, updated_at=? WHERE id=?", (gid, now_str, pid))
                if tags_str:
                    raw_tags = [t.strip() for t in tags_str.split(',') if t.strip()]
                    for tag_name in raw_tags:
                        try:
                            conn.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
                        except sqlite3.IntegrityError:
                            pass
                        tag_id = conn.execute("SELECT id FROM tags WHERE name=?", (tag_name,)).fetchone()[0]
                        try:
                            conn.execute("INSERT INTO prompt_tags (prompt_id, tag_id) VALUES (?, ?)", (int(pid), tag_id))
                        except sqlite3.IntegrityError:
                            pass
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
            flash("Backup restored.")
            return redirect(url_for("manage"))
        except:
            flash("Restore failed.")
            return redirect(url_for("restore_backup"))
    elif filename.endswith(".db") or filename.endswith(".sqlite"):
        try:
            f.save(DB_PATH)
            flash("Database restored.")
            return redirect(url_for("manage"))
        except:
            pass

    return redirect(url_for("restore_backup"))


# ---------------------------
# Ollama Integration
# ---------------------------
OLLAMA_DEFAULT_MODEL = "gemma3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"


# Cached readiness detection (prevents hammering /api/tags)
_OLLAMA_CACHE = {"ts": 0.0, "ready": False, "models": []}

def ollama_models_cached(ttl_seconds: float = 10.0) -> list[str]:
    now = time.time()
    if (now - _OLLAMA_CACHE["ts"]) < ttl_seconds:
        return list(_OLLAMA_CACHE.get("models") or [])
    models = ollama_list_models()
    _OLLAMA_CACHE["ts"] = now
    _OLLAMA_CACHE["models"] = models
    _OLLAMA_CACHE["ready"] = bool(models)
    return list(models)

def ollama_is_ready_cached(ttl_seconds: float = 10.0) -> bool:
    return bool(ollama_models_cached(ttl_seconds=ttl_seconds))


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
    payload = {
        "model": model or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    if images:
        payload["images"] = images

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

    save_action = request.form.get("save_action")
    use_render = (request.form.get("use_render") or "").strip() == "1"
    if use_render:
        # Re-render from immutable base + fills to avoid relying on client-side JS.
        base_content = prompt["content"]
        placeholders_meta = extract_placeholders_meta(base_content)
        fills = parse_use_fills_from_form(request.form, placeholders_meta)

        override_active = (request.form.get("override_active") or "").strip() == "1"
        override_text = (request.form.get("final_override") or "").strip() if override_active else None

        current_content = render_use_output(base_content, fills, final_override=override_text)
    else:
        current_content = request.form.get("content") or prompt["content"]

    if not save_action:
        if not ollama_available:
            flash("Ollama not available.")
            return redirect(url_for("render_prompt", prompt_id=prompt_id))

        model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
        session["ollama_model"] = model
        instruction = (request.form.get("instruction") or "").strip()
        mode = request.form.get("mode") or "refine"

        try:
            refined = ollama_generate(
                f"MODE: {mode}\nINSTRUCTION: {instruction}\n\nPROMPT:\n{current_content}",
                model=model,
                system="Output only refined prompt."
            )
            flash("Refinement generated. Review below.")
        except Exception as e:
            flash(f"Ollama error: {e}")
            refined = current_content

        p_preview = dict(prompt)
        p_preview['content'] = refined
        return render_template("refine_prompt.html", prompt=p_preview, models=models, ollama_available=ollama_available)

    else:
        if save_action == "overwrite":
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "UPDATE prompts SET content = ?, updated_at = ? WHERE id = ?",
                    (current_content, datetime.utcnow().isoformat(), prompt_id)
                )
                conn.commit()
            flash("Prompt updated.")
            return redirect(url_for("render_prompt", prompt_id=prompt_id))

        elif save_action == "variant":
            parent_id = prompt["parent_id"] if prompt["parent_id"] else prompt["id"]
            new_title = request.form.get("new_title") or f"{prompt['title']} (Refined)"

            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, group_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_title,
                        prompt["category"],
                        prompt["tool"],
                        prompt["prompt_type"],
                        current_content,
                        prompt["notes"],
                        prompt["thumbnail"],
                        parent_id,
                        (prompt["group_id"] if ("group_id" in prompt.keys()) else None),
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                    )
                )
                conn.commit()
            flash("Saved as new variant.")
            return redirect(url_for("index", view_family=parent_id))

        elif save_action == "new":
            new_title = request.form.get("new_title") or f"{prompt['title']} (Refined)"
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, group_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_title,
                        prompt["category"],
                        prompt["tool"],
                        prompt["prompt_type"],
                        current_content,
                        prompt["notes"],
                        prompt["thumbnail"],
                        None,
                        (prompt["group_id"] if ("group_id" in prompt.keys()) else None),
                        datetime.utcnow().isoformat(),
                        datetime.utcnow().isoformat(),
                    ),
                )
                conn.commit()
            flash("Saved as new independent prompt.")
            return redirect(url_for("index"))


@app.route("/prompt/<int:prompt_id>/save_from_use", methods=["POST"])
def save_from_use(prompt_id: int):
    """Save edited prompt text directly from the Use screen."""
    init_db()
    prompt = get_prompt(prompt_id)
    if not prompt:
        flash("Prompt not found.")
        return redirect(url_for("index"))

    mode = request.form.get("save_mode", "overwrite")
    use_render = (request.form.get("use_render") or "").strip() == "1"
    if use_render:
        base_content = prompt["content"]
        placeholders_meta = extract_placeholders_meta(base_content)
        fills = parse_use_fills_from_form(request.form, placeholders_meta)

        override_active = (request.form.get("override_active") or "").strip() == "1"
        override_text = (request.form.get("final_override") or "").strip() if override_active else None

        content = render_use_output(base_content, fills, final_override=override_text)
    else:
        content = (request.form.get("content") or "").strip()
    if not content:
        flash("Nothing to save.")
        return redirect(url_for("render_prompt", prompt_id=prompt_id))

    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        if mode == "variant":
            parent_id = prompt["parent_id"] if prompt["parent_id"] else prompt["id"]
            new_title = f"{prompt['title']} (Variant)"
            conn.execute(
                """
                INSERT INTO prompts (title, category, tool, prompt_type, content, notes, thumbnail, parent_id, group_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (new_title, prompt["category"], prompt["tool"], prompt["prompt_type"], content, prompt["notes"], prompt["thumbnail"], parent_id, (prompt["group_id"] if ("group_id" in prompt.keys()) else None), now, now),
            )
            new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            # Copy tags to the variant
            tags = get_tags_for_prompt(conn, prompt_id)
            save_tags(conn, new_id, ", ".join(tags))
            conn.commit()
            flash("Saved as variant.")
            return redirect(url_for("index", view_family=parent_id))

        # overwrite
        conn.execute(
            "UPDATE prompts SET content = ?, updated_at = ? WHERE id = ?",
            (content, now, prompt_id),
        )
        conn.commit()

    flash("Prompt updated.")
    return redirect(url_for("render_prompt", prompt_id=prompt_id))


@app.route("/api/prompt/<int:prompt_id>/render_final", methods=["POST"])
def api_render_final(prompt_id: int):
    """Server-side final renderer for Use Prompt (copy/export consistency)."""
    init_db()
    prompt = get_prompt(prompt_id)
    if not prompt:
        return jsonify({"error": "Prompt not found."}), 404

    # NSFW direct access protection (match Use Prompt view)
    if NSFW_LOCK_ENABLED_DEFAULT and not nsfw_is_unlocked():
        with get_db() as conn:
            if prompt_is_nsfw(conn, prompt_id):
                return jsonify({"error": "This prompt is locked (NSFW)."}), 403

    base_content = prompt["content"]
    placeholders_meta = extract_placeholders_meta(base_content)
    fills = parse_use_fills_from_form(request.form, placeholders_meta)

    override_active = (request.form.get("override_active") or "").strip() == "1"
    override_text = (request.form.get("final_override") or "").strip() if override_active else None

    final_text = render_use_output(base_content, fills, final_override=override_text)
    return jsonify({"text": final_text})


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
        return render_template(
            "edit_prompt.html",
            prompt=prompt_like,
            categories=categories,
            tools=tools,
            prompt_types=PROMPT_TYPES,
            refine_error="Ollama unavailable",
            ollama_available=False,
            tags_str=""
        )

    model = request.form.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    session["ollama_model"] = model
    try:
        refined = ollama_generate(
            f"MODE: {request.form.get('mode')}\nINSTRUCTION: {request.form.get('instruction')}\n\nPROMPT:\n{base}",
            model=model,
            system="Output only refined prompt."
        )
        return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refined_draft=refined, ollama_available=True, tags_str="")
    except Exception as e:
        return render_template("edit_prompt.html", prompt=prompt_like, categories=categories, tools=tools, prompt_types=PROMPT_TYPES, refine_error=str(e), ollama_available=True, tags_str="")

# Vision → draft prompt (image) endpoint (updated for template compatibility)
@app.route("/api/generate_draft_from_image", methods=["POST"])
def generate_draft_from_image():
    # Accept either key just in case
    f = request.files.get("image") or request.files.get("file")

    vision_model = (request.form.get("vision_model") or "").strip()
    text_model = (request.form.get("text_model") or "").strip()

    # Match the field names used by the current edit_prompt.html
    vision_instruction = (request.form.get("vision_instruction") or "").strip()
    draft_instruction = (request.form.get("draft_instruction") or "").strip()

    if not f or not vision_model or not text_model:
        return jsonify({"error": "Missing image or model selection."}), 400

    try:
        thumbnail_path = save_thumbnail(f)
        saved_file_path = BASE_DIR / "static" / thumbnail_path
        with open(saved_file_path, "rb") as image_file:
            img_b64 = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 400

    vision_sys = (
        "You are a visual analysis model. Describe the contents of the provided image clearly and factually. "
        "Focus on Subject, Appearance, Environment, Composition, Lighting, Mood."
    )

    vision_prompt = "Describe this image."
    if vision_instruction:
        vision_prompt += " " + vision_instruction

    try:
        vision_response = ollama_generate(
            prompt=vision_prompt,
            model=vision_model,
            system=vision_sys,
            images=[img_b64],
        )
    except Exception as e:
        return jsonify({"error": f"Vision model failed: {str(e)}"}), 500

    draft_sys = (
        "You are assisting with AI prompt creation. Based on the visual description, generate a draft AI image prompt "
        "suitable for models like Flux or Stable Diffusion. Output a single paragraph."
    )

    draft_user_prompt = "Visual description:\n" + str(vision_response)
    if draft_instruction:
        draft_user_prompt += "\n\nAdditional Instructions:\n" + draft_instruction

    try:
        draft_response = ollama_generate(
            prompt=draft_user_prompt,
            model=text_model,
            system=draft_sys,
        )
    except Exception as e:
        return jsonify({"error": f"Text model failed: {str(e)}"}), 500

    # Return keys that the current edit_prompt.html expects:
    # - content: goes into the main prompt textarea
    # - thumbnail_url: updates the thumbnail preview
    # - thumbnail_path: stored in hidden field for later save
    return jsonify({
        "success": True,
        "description": vision_response,
        "draft_prompt": draft_response,
        "content": draft_response,
        "thumbnail_path": thumbnail_path,
        "thumbnail_url": url_for("static", filename=thumbnail_path),
    })


# Backwards-compatible alias for older/newer templates
@app.route("/api/generate_prompt_from_image", methods=["POST"])
def generate_prompt_from_image_alias():
    return generate_draft_from_image()


@app.route("/prompt/<int:prompt_id>/raw", methods=["GET"])
def prompt_raw(prompt_id: int):
    with get_db() as conn:
        p = conn.execute("SELECT * FROM prompts WHERE id = ?", (prompt_id,)).fetchone()
        if not p:
            return jsonify({"error": "not found"}), 404

        # --- NSFW direct access protection ---
        if NSFW_LOCK_ENABLED_DEFAULT and not nsfw_is_unlocked():
            if prompt_is_nsfw(conn, prompt_id):
                return jsonify({"error": "locked"}), 403

        return jsonify({
            "id": p["id"],
            "title": p["title"],
            "content": p["content"] or "",
            "category": p["category"],
            "tool": p["tool"],
            "prompt_type": p["prompt_type"],
        })

@app.route("/api/generate_refinement", methods=["POST"])
def api_generate_refinement():
    data = request.get_json()
    base_content = data.get("content", "")
    model = data.get("model") or session.get("ollama_model") or OLLAMA_DEFAULT_MODEL
    instruction = data.get("instruction", "")
    system_role = data.get("system_role", "").strip()

    if not base_content:
        return jsonify({"error": "Content field is empty."}), 400

    if not system_role:
        system_role = "You are an expert prompt engineer. Refine the following prompt to be more descriptive and effective. Output ONLY the refined prompt text, no preamble."

    full_prompt = f"INSTRUCTION: {instruction}\n\nORIGINAL PROMPT:\n{base_content}"

    try:
        refined = ollama_generate(prompt=full_prompt, model=model, system=system_role)
        return jsonify({"success": True, "content": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/api/prompt/<int:prompt_id>/usage", methods=["GET", "POST"])
def api_prompt_usage(prompt_id):
    init_db()
    with get_db() as conn:
        p = conn.execute(
            "SELECT id FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if not p:
            return jsonify({"ok": False, "error": "not found"}), 404

        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            content = (data.get("content") or "").strip()
            source = (data.get("source") or "clipboard").strip()

            if not content:
                return jsonify({"ok": False, "error": "empty"}), 400

            last = conn.execute(
                "SELECT content FROM prompt_uses WHERE prompt_id = ? ORDER BY created_at DESC LIMIT 1",
                (prompt_id,),
            ).fetchone()

            if last and (last["content"] or "") == content:
                return jsonify({"ok": True, "deduped": True})

            now = datetime.utcnow().isoformat(timespec="seconds")
            conn.execute(
                "INSERT INTO prompt_uses (prompt_id, content, created_at, source) VALUES (?,?,?,?)",
                (prompt_id, content, now, source),
            )

            conn.execute(
                """
                DELETE FROM prompt_uses
                WHERE id IN (
                    SELECT id FROM prompt_uses
                    WHERE prompt_id = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET 250
                )
                """,
                (prompt_id,),
            )

            conn.commit()
            return jsonify({"ok": True})

        rows = conn.execute(
            "SELECT id, content, created_at, source FROM prompt_uses WHERE prompt_id = ? ORDER BY created_at DESC LIMIT 20",
            (prompt_id,),
        ).fetchall()

        return jsonify([dict(r) for r in rows])


@app.route("/prompt/<int:prompt_id>/history")
def prompt_history(prompt_id):
    init_db()
    with get_db() as conn:
        prompt = conn.execute(
            "SELECT * FROM prompts WHERE id = ?",
            (prompt_id,),
        ).fetchone()
        if not prompt:
            flash("Prompt not found")
            return redirect(url_for("index"))

        rows = conn.execute(
            "SELECT * FROM prompt_uses WHERE prompt_id = ? ORDER BY created_at DESC",
            (prompt_id,),
        ).fetchall()

    return render_template(
        "prompt_history.html",
        prompt=prompt,
        uses=rows,
    )


# --- MAIN BLOCK MUST BE LAST ---
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
