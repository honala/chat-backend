@echo off
setlocal

REM Manual debug: keeps this window open and shows uvicorn logs.
REM No black box at logon: run install-autostart.ps1 once (pythonw + hidden PowerShell task).

REM Ensure we run from this folder
cd /d "%~dp0"

REM Separate tokens:
REM - User upload token (shareable with friends)
REM - Admin token (for admin.html delete/prune/list)
set "HONALA_UPLOAD_TOKEN_USER=4829572"
set "HONALA_UPLOAD_TOKEN_ADMIN=xxp071018@"

REM Optional overrides (uncomment if you want)
REM set "HONALA_PHOTOS_ROOT=C:\Users\XXPxx\Desktop\nginx-1.29.5\html\photos"
REM set "HONALA_UPLOAD_MAX_BYTES=31457280"
REM set "HONALA_THUMB_MAX_W=640"
REM set "HONALA_THUMB_QUALITY=82"

python -m uvicorn server:app --host 127.0.0.1 --port 5055

