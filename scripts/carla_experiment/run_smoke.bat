@echo off
setlocal
set "SCRIPT=%~dp0carla_fca_recalibration.py"
if not defined VIRTUAL_ENV call "venv37\Scripts\activate"
echo Quick smoke test: 1 seed, short episode, Town03 ...
python "%SCRIPT%" --town Town03 --seeds 1 --episode-len 1200 --arms all --out "%~dp0carla_results_smoke.csv"
echo.
pause
endlocal
