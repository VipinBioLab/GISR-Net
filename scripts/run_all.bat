@echo off
REM ==========================================================================
REM  GISR-Net revision -- run every experiment.  (Windows)
REM
REM    cd  path\to\GISRNET_Work\GISRNet_PyTorch
REM    run_all.bat
REM
REM  A CUDA GPU is used automatically when available, otherwise the CPU.
REM  The runner is RESUMABLE: train.py skips folds already recorded in
REM  results_<tag>.json, so if you stop it (Ctrl-C) and start it again it
REM  continues from where it left off.
REM
REM  Optional environment overrides, set BEFORE calling the script:
REM      set EPOCHS=10        shorter trial run          (default 30)
REM      set ACCUM=4          micro-batch 8 for low RAM  (default 1)
REM      set WORKERS=0        data-loader processes      (default 0)
REM      set PY=py -3.11      pick a specific interpreter
REM ==========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0..\src"

if not defined PY      set "PY=python"
if not defined EPOCHS  set "EPOCHS=30"
if not defined ACCUM   set "ACCUM=1"
if not defined WORKERS set "WORKERS=0"

echo.
echo === checking dependencies ===
%PY% -c "import importlib.util,sys; m=[p for p in ('torch','torchvision','numpy','scipy','matplotlib','pandas','PIL','docx') if importlib.util.find_spec(p) is None]; sys.exit(0) if not m else (print('Missing packages: '+', '.join(m)), print('Install with:  pip install torch torchvision numpy scipy matplotlib pandas pillow python-docx'), sys.exit(1))"
if errorlevel 1 goto :fail

%PY% -c "import torch; print('torch', torch.__version__, ' device=', 'cuda' if torch.cuda.is_available() else 'cpu')"
if errorlevel 1 goto :fail

for /f "delims=" %%R in ('%PY% -c "import config;print(config.OUT_DIR)"') do set "RES=%%R"
set "LOG=%RES%\logs"
if not exist "%LOG%" mkdir "%LOG%"

echo.
echo === step 0: fold definition and leakage check ===
%PY% folds.py
if errorlevel 1 goto :fail

echo.
echo === step 1: external test sets (HF_Test + MED-NODE) ===
%PY% build_external_sets.py

echo.
echo === step 2: train GISR-Net ===
call :run gisrnet_tl --arch gisrnet

echo.
echo === step 3: ablation study ===
call :run gisrnet_ri      --arch gisrnet --no-pretrained
call :run gisrnet_noaug   --arch gisrnet --no-augment
call :run gisrnet_mlp     --arch gisrnet --head mlp
call :run gisrnet_sigmoid --arch gisrnet --head sigmoid

echo.
echo === step 4: generalisability on the external sets ===
%PY% evaluate_external.py --tags gisrnet_tl

echo.
echo === step 5: figures, tables and documents ===
%PY% make_figures.py
%PY% make_tables.py
%PY% make_manuscript.py

echo.
echo ALL RUNS COMPLETE
echo Results are in: %RES%
endlocal
exit /b 0

REM --------------------------------------------------------------------------
:run
set "TAG=%~1"
shift
set "ARGS="
:collect
if "%~1"=="" goto :gotargs
set "ARGS=!ARGS! %~1"
shift
goto :collect
:gotargs
echo.
echo [%TIME:~0,8%] ^>^>^> %TAG%
REM --auto-lr applies the SAME learning-rate selection procedure to every
REM architecture, using the fold-1 validation partition only.
%PY% -W ignore train.py --tag %TAG% --workers %WORKERS% --auto-lr --accum %ACCUM% --epochs %EPOCHS%!ARGS!
echo [%TIME:~0,8%] ^<^<^< %TAG% done
exit /b 0

REM --------------------------------------------------------------------------
:fail
echo.
echo Setup check failed -- nothing was run.
endlocal
exit /b 1
