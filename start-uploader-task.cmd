@echo off
cd /d "C:\Users\XXPxx\Desktop\nginx-1.29.5\photos-uploader"
set HONALA_UPLOAD_TOKEN_USER=4829572
set HONALA_UPLOAD_TOKEN_ADMIN=xxp071018@

rem Start uploader detached/hidden, then exit this cmd quickly.
powershell -NoProfile -WindowStyle Hidden -Command ^
  "$env:HONALA_UPLOAD_TOKEN_USER='4829572';" ^
  "$env:HONALA_UPLOAD_TOKEN_ADMIN='xxp071018@';" ^
  "Start-Process -FilePath 'py.exe' -ArgumentList @('-3','-m','uvicorn','server:app','--host','127.0.0.1','--port','5055') -WorkingDirectory 'C:\Users\XXPxx\Desktop\nginx-1.29.5\photos-uploader' -WindowStyle Hidden"

exit /b 0
