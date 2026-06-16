@echo off
setlocal enabledelayedexpansion

REM ============= CONFIG -- edit if needed =============
REM If you run this from an already-activated venv37, VENV is ignored.
set "VENV=venv37"
set "SCRIPT=%~dp0carla_fca_recalibration.py"
set "TOWNS=Town03 Town05"
set "SEEDS=10"
set "EPLEN=2400"
set "HOST=127.0.0.1"
set "PORT=2000"
REM ===================================================

if defined VIRTUAL_ENV (
  echo Using already-active venv: %VIRTUAL_ENV%
) else (
  echo Activating venv: %VENV%
  call "%VENV%\Scripts\activate"
)

echo.
echo Checking CARLA server at %HOST%:%PORT% ...
python -c "import carla; c=carla.Client('%HOST%',%PORT%); c.set_timeout(60.0); print('CARLA server', c.get_server_version())" 2>nul
if errorlevel 1 (
  echo.
  echo [ERROR] CARLA server not reachable.
  echo Start it first:  CarlaUE4.exe -quality-level=Epic -carla-rpc-port=%PORT%
  echo Wait until the 3D city window appears, then run this script again.
  goto :done
)

for %%T in (%TOWNS%) do (
  echo.
  echo ===== Running %%T : seeds=%SEEDS%  episode-len=%EPLEN% =====
  python "%SCRIPT%" --host %HOST% --port %PORT% --town %%T --seeds %SEEDS% --episode-len %EPLEN% --arms all --out "%~dp0carla_results_%%T.csv"
)

echo.
echo ===== Summarizing all results =====
python "%~dp0analyze_results.py"

:done
echo.
echo Done.
pause
endlocal
