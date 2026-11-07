@echo off
setlocal
REM dao-fa-grader one-click launcher: service + workbench (frontend)
REM Usage: start.bat [port]   (default 8000)
cd /d "%~dp0"

set PORT=8000
if not "%~1"=="" set PORT=%~1

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

set CURL=%SystemRoot%\System32\curl.exe
set TIMEOUT=%SystemRoot%\System32\timeout.exe

%PY% -c "import grader, uvicorn" 2>nul
if errorlevel 1 (
  echo [ERROR] grader is not installed. Run:  python -m pip install -e ".[dev]"
  pause
  exit /b 1
)

REM Already our healthy service on this port? Just open the workbench.
%CURL% -s -o nul http://127.0.0.1:%PORT%/healthz
if not errorlevel 1 (
  echo Service already running on port %PORT%. Opening workbench ...
  start "" "http://127.0.0.1:%PORT%/workbench/"
  exit /b 0
)

REM Port occupied by another process?
%SystemRoot%\System32\netstat.exe -ano | %SystemRoot%\System32\findstr.exe ":%PORT%" | %SystemRoot%\System32\findstr.exe "LISTENING" >nul
if not errorlevel 1 (
  echo [ERROR] Port %PORT% is occupied by another process.
  echo         Run:  stop.bat %PORT%   ^(or choose another port: start.bat 9000^)
  pause
  exit /b 1
)

echo Starting dao-fa-grader on port %PORT% ...
start "dao-fa-grader (port %PORT%)" cmd /k "%PY%" -m uvicorn grader.main:app --host 127.0.0.1 --port %PORT%

set /a tries=0
:waitloop
%TIMEOUT% /t 1 /nobreak >nul
%CURL% -s -o nul http://127.0.0.1:%PORT%/healthz
if errorlevel 1 (
  set /a tries+=1
  if %tries% lss 30 goto waitloop
  echo [ERROR] service not healthy after 30s. Check the server window for details.
  pause
  exit /b 1
)

echo Service is healthy. Opening workbench in browser ...
start "" "http://127.0.0.1:%PORT%/workbench/"
echo Ready. Close the server window to stop, or run:  stop.bat %PORT%
%TIMEOUT% /t 3 /nobreak >nul
