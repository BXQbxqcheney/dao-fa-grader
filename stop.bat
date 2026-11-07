@echo off
setlocal enabledelayedexpansion
REM dao-fa-grader stopper: kill the process listening on [port] (default 8000)
REM Usage: stop.bat [port]
set PORT=8000
if not "%~1"=="" set PORT=%~1

set PID=
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set PID=%%p

if "%PID%"=="" (
  echo No process is listening on port %PORT%. Nothing to stop.
  exit /b 0
)

echo Stopping dao-fa-grader (PID %PID%, port %PORT%) ...
taskkill /PID %PID% /F
