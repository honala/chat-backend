import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return default if v is None else v


PHOTOS_ROOT = Path(_env("HONALA_PHOTOS_ROOT", "/var/www/html/photos")).resolve()
FILES_DIR = (PHOTOS_ROOT / "files").resolve()
THUMBS_DIR = (FILES_DIR / "thumbs").resolve()
META_PATH = (PHOTOS_ROOT / "meta.json").resolve()
UPLOAD_TOKEN = _env("HONALA_UPLOAD_TOKEN_USER", "CHANGE_ME")
ADMIN_TOKEN = _env("HONALA_UPLOAD_TOKEN_ADMIN", "CHANGE_ME")

MAX_BYTES = int(_env("HONALA_UPLOAD_MAX_BYTES", "31457280"))  # 30 MiB
THUMB_MAX_W = int(_env("HONALA_THUMB_MAX_W", "640"))
THUMB_QUALITY = int(_env("HONALA_THUMB_QUALITY", "82"))

SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif"}


app = FastAPI(title="Honala Photos Uploader", version="1.0")


def safe_filename(name: str) -> str:
    base = Path(name).name
    base = base.replace(" ", "_")
    base = SAFE_NAME_RE.sub("_", base).strip("._-")
    if not base:
        base = "photo"
    ext = Path(base).suffix.lower()
    if ext not in ALLOWED_EXT:
        # default to jpg if unknown; original bytes stay, but we reject unknown later
        pass
    return base


def utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_meta() -> dict:
    if not META_PATH.exists():
        return {"version": 1, "photos": {}}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "photos": {}}


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def ensure_dirs() -> None:
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def list_original_files() -> set[str]:
    if not FILES_DIR.exists():
        return set()
    out: set[str] = set()
    for p in FILES_DIR.iterdir():
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in ALLOWED_EXT:
            out.add(p.name)
    return out


def require_admin(token: str):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return None


def require_upload(token: str):
    if not UPLOAD_TOKEN or token != UPLOAD_TOKEN:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    return None


def make_thumb(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        w, h = im.size
        if w > THUMB_MAX_W:
            new_h = int(h * (THUMB_MAX_W / w))
            im = im.resize((THUMB_MAX_W, max(1, new_h)), Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_path, format="JPEG", quality=THUMB_QUALITY, optimize=True, progressive=True)


@app.get("/health")
def health():
    return {
        "ok": True,
        "photos_root": str(PHOTOS_ROOT),
        "files_dir": str(FILES_DIR),
        "thumbs_dir": str(THUMBS_DIR),
    }

@app.post("/list")
async def list_items(token: str = Form(...)):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    existing = list_original_files()
    meta = load_meta()
    photos = meta.get("photos") if isinstance(meta, dict) else None
    if not isinstance(photos, dict):
        photos = {}

    items = []
    for name in sorted(existing):
        m = photos.get(name) if isinstance(photos, dict) else None
        hidden = bool(isinstance(m, dict) and m.get("hidden") is True)
        items.append(
            {
                "file": name,
                "title": (m or {}).get("title") if isinstance(m, dict) else None,
                "author": (m or {}).get("author") if isinstance(m, dict) else None,
                "caption": (m or {}).get("caption") if isinstance(m, dict) else None,
                "story": (m or {}).get("story") if isinstance(m, dict) else None,
                "uploadedAt": (m or {}).get("uploadedAt") if isinstance(m, dict) else None,
                "hidden": hidden,
            }
        )

    return {"ok": True, "count": len(items), "items": items}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    token: str = Form(...),
    author: str = Form(""),
    title: str = Form(""),
    caption: str = Form(""),
    story: str = Form(""),
):
    err = require_upload(token)
    if err is not None:
        return err

    ensure_dirs()

    original_name = file.filename or "photo"
    safe_name = safe_filename(original_name)
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": f"unsupported file type: {ext}"}, status_code=400)

    # Avoid overwrite: add timestamp suffix if exists
    target = (FILES_DIR / safe_name).resolve()
    if target.exists():
        stem = Path(safe_name).stem
        target = (FILES_DIR / f"{stem}-{int(datetime.now().timestamp())}{ext}").resolve()

    # Read with size limit
    size = 0
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_BYTES:
                try:
                    target.unlink(missing_ok=True)
                except Exception:
                    pass
                return JSONResponse({"error": "file too large"}, status_code=413)
            out.write(chunk)

    # Create thumb (jpg). Keep same filename for mapping; always .jpg for thumbs
    thumb_name = Path(target.name).with_suffix(".jpg").name
    thumb_path = (THUMBS_DIR / thumb_name).resolve()
    try:
        make_thumb(target, thumb_path)
    except Exception as e:
        # If thumb fails, still keep original
        return JSONResponse({"error": f"thumbnail failed: {e}"}, status_code=500)

    # Update meta.json mapping by original filename; for thumbs, we keep same base and let frontend request thumbs/<original>
    # To align, if original isn't .jpg, also write a same-name thumb copy reference via symlink-like naming:
    if Path(target.name).suffix.lower() != ".jpg":
        # Create a duplicate file name to satisfy frontend thumbs/<originalName>
        alt_thumb_path = (THUMBS_DIR / target.name).resolve()
        try:
            # Copy bytes
            alt_thumb_path.write_bytes(thumb_path.read_bytes())
        except Exception:
            pass

    meta = load_meta()
    if not isinstance(meta, dict):
        meta = {"version": 1, "photos": {}}
    meta.setdefault("version", 1)
    meta.setdefault("photos", {})
    if not isinstance(meta["photos"], dict):
        meta["photos"] = {}

    meta["photos"][target.name] = {
        "title": title.strip() or Path(target.name).stem,
        "author": author.strip() or "Photography",
        "caption": caption.strip() or "Uploaded",
        "story": (story or "").replace("\r\n", "\n").rstrip(),
        "uploadedAt": utc_iso(),
        "size": size,
        "hidden": False,
    }

    atomic_write(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    return {"ok": True, "file": target.name, "thumb": f"thumbs/{target.name}"}


@app.post("/delete")
async def delete(
    token: str = Form(...),
    filename: str = Form(...),
):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    name = safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": "unsupported file type"}, status_code=400)

    target = (FILES_DIR / name).resolve()
    # Safety: ensure file is inside FILES_DIR
    if target.parent != FILES_DIR:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    # Delete original
    existed = target.exists()
    target.unlink(missing_ok=True)

    # Delete thumb variants
    (THUMBS_DIR / name).resolve().unlink(missing_ok=True)
    (THUMBS_DIR / Path(name).with_suffix(".jpg").name).resolve().unlink(missing_ok=True)

    # Remove meta entry if present
    meta = load_meta()
    photos = meta.get("photos") if isinstance(meta, dict) else None
    if isinstance(photos, dict) and name in photos:
        photos.pop(name, None)
        atomic_write(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    return {"ok": True, "deleted": name, "existed": existed}


@app.post("/prune")
async def prune(token: str = Form(...)):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    existing = list_original_files()

    meta = load_meta()
    if not isinstance(meta, dict):
        meta = {"version": 1, "photos": {}}
    meta.setdefault("version", 1)
    meta.setdefault("photos", {})
    if not isinstance(meta["photos"], dict):
        meta["photos"] = {}

    photos: dict = meta["photos"]
    before = len(photos)
    for k in list(photos.keys()):
        if k not in existing:
            photos.pop(k, None)
    after = len(photos)

    atomic_write(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    return {"ok": True, "before": before, "after": after, "removed": before - after}


@app.post("/meta-update")
async def meta_update(
    token: str = Form(...),
    filename: str = Form(...),
    author: str = Form(""),
    title: str = Form(""),
    caption: str = Form(""),
    story: str = Form(""),
):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    name = safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": "unsupported file type"}, status_code=400)

    target = (FILES_DIR / name).resolve()
    if target.parent != FILES_DIR:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    if not target.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)

    meta = load_meta()
    if not isinstance(meta, dict):
        meta = {"version": 1, "photos": {}}
    meta.setdefault("version", 1)
    meta.setdefault("photos", {})
    if not isinstance(meta["photos"], dict):
        meta["photos"] = {}

    photos: dict = meta["photos"]
    cur = photos.get(name)
    if not isinstance(cur, dict):
        cur = {}
    prev_hidden = bool(cur.get("hidden") is True)
    try:
        st = target.stat()
    except OSError:
        return JSONResponse({"error": "file not readable"}, status_code=500)

    cur.setdefault("uploadedAt", utc_iso())
    cur["size"] = st.st_size
    cur["author"] = author.strip() or cur.get("author") or "Photography"
    cur["title"] = title.strip() or cur.get("title") or Path(name).stem
    cur["caption"] = caption.strip() or cur.get("caption", "Uploaded")
    cur["story"] = (story or "").replace("\r\n", "\n").rstrip()
    cur["hidden"] = prev_hidden

    photos[name] = cur
    atomic_write(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    return {"ok": True, "file": name}


@app.post("/set-visibility")
async def set_visibility(
    token: str = Form(...),
    filename: str = Form(...),
    hidden: str = Form("false"),
):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    name = safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": "unsupported file type"}, status_code=400)

    target = (FILES_DIR / name).resolve()
    if target.parent != FILES_DIR:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    if not target.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)

    is_hidden = str(hidden).lower() in ("1", "true", "yes", "on")

    meta = load_meta()
    if not isinstance(meta, dict):
        meta = {"version": 1, "photos": {}}
    meta.setdefault("version", 1)
    meta.setdefault("photos", {})
    if not isinstance(meta["photos"], dict):
        meta["photos"] = {}

    photos: dict = meta["photos"]
    cur = photos.get(name)
    if not isinstance(cur, dict):
        cur = {}
        try:
            st = target.stat()
        except OSError:
            return JSONResponse({"error": "file not readable"}, status_code=500)
        cur["title"] = Path(name).stem
        cur["author"] = "Photography"
        cur["caption"] = "Uploaded"
        cur["story"] = ""
        cur["uploadedAt"] = utc_iso()
        cur["size"] = st.st_size

    cur["hidden"] = is_hidden
    photos[name] = cur
    atomic_write(META_PATH, json.dumps(meta, ensure_ascii=False, indent=2) + "\n")

    return {"ok": True, "file": name, "hidden": is_hidden}


@app.post("/thumb-refresh")
async def thumb_refresh(token: str = Form(...), filename: str = Form(...)):
    err = require_admin(token)
    if err is not None:
        return err

    ensure_dirs()
    name = safe_filename(filename)
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return JSONResponse({"error": "unsupported file type"}, status_code=400)

    target = (FILES_DIR / name).resolve()
    if target.parent != FILES_DIR:
        return JSONResponse({"error": "invalid path"}, status_code=400)

    if not target.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)

    thumb_name = Path(target.name).with_suffix(".jpg").name
    thumb_path = (THUMBS_DIR / thumb_name).resolve()
    try:
        make_thumb(target, thumb_path)
    except Exception as e:
        return JSONResponse({"error": f"thumbnail failed: {e}"}, status_code=500)

    if Path(target.name).suffix.lower() != ".jpg":
        alt_thumb_path = (THUMBS_DIR / target.name).resolve()
        try:
            alt_thumb_path.write_bytes(thumb_path.read_bytes())
        except Exception:
            pass

    return {"ok": True, "file": name}

