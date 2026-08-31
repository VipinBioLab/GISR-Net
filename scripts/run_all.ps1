# ===========================================================================
#  GISR-Net revision -- run every experiment.  (Windows PowerShell)
#
#    cd path\to\GISRNET_Work\GISRNet_Windows
#    powershell -ExecutionPolicy Bypass -File run_all.ps1
#
#  A CUDA GPU is used automatically when available, otherwise the CPU.
#  RESUMABLE: train.py skips folds already recorded in results_<tag>.json,
#  so stopping with Ctrl-C and re-running continues from where it left off.
#
#  Optional parameters:
#    -Epochs 10      shorter trial run             (default 30)
#    -Accum 4        micro-batch 8 for low RAM     (default 1)
#    -Workers 0      data-loader processes         (default 0)
#    -Py "py -3.11"  choose a specific interpreter (default "python")
# ===========================================================================

param(
    [int]$Epochs  = 30,
    [int]$Accum   = 1,
    [int]$Workers = 0,
    [string]$Py   = "python"
)

$ErrorActionPreference = "Continue"
Set-Location -Path (Join-Path $PSScriptRoot ".." | Join-Path -ChildPath "src")

function Invoke-Py {
    param([string[]]$PyArgs)
    $exe, $pre = ($Py -split '\s+')[0], ($Py -split '\s+' | Select-Object -Skip 1)
    & $exe @($pre + $PyArgs)
    return $LASTEXITCODE
}

Write-Host ""
Write-Host "=== checking dependencies ===" -ForegroundColor Cyan
$check = @'
import importlib.util, sys
missing = [m for m in ("torch", "torchvision", "numpy", "scipy",
                       "matplotlib", "pandas", "PIL", "docx")
           if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages: " + ", ".join(missing))
    print("Install with:  pip install torch torchvision numpy scipy matplotlib pandas pillow python-docx")
    sys.exit(1)
import torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__}  device={dev}")
'@
$check | & (($Py -split '\s+')[0]) @(($Py -split '\s+' | Select-Object -Skip 1) + @("-"))
if ($LASTEXITCODE -ne 0) {
    Write-Host "Setup check failed -- nothing was run." -ForegroundColor Red
    exit 1
}

$RES = (Invoke-Expression "$Py -c ""import config;print(config.OUT_DIR)""")
$LOG = Join-Path $RES "logs"
New-Item -ItemType Directory -Force -Path $LOG | Out-Null

function Run-Config {
    param([string]$Tag, [string[]]$Extra)
    Write-Host ""
    Write-Host ("[{0}] >>> {1}" -f (Get-Date -Format HH:mm:ss), $Tag) -ForegroundColor Green
    $logFile = Join-Path $LOG "$Tag.log"
    # --auto-lr applies the SAME learning-rate selection procedure to every
    # architecture, using the fold-1 validation partition only.
    Invoke-Py (@("-W", "ignore", "train.py", "--tag", $Tag, "--auto-lr",
                 "--workers", $Workers, "--accum", $Accum,
                 "--epochs", $Epochs) + $Extra) 2>&1 | Tee-Object -FilePath $logFile -Append
    Write-Host ("[{0}] <<< {1} done" -f (Get-Date -Format HH:mm:ss), $Tag) -ForegroundColor Green
}

Write-Host ""
Write-Host "=== step 0: fold definition and leakage check ===" -ForegroundColor Cyan
if ((Invoke-Py @("folds.py")) -ne 0) { exit 1 }

Write-Host ""
Write-Host "=== step 1: external test sets (HF_Test + MED-NODE) ===" -ForegroundColor Cyan
Invoke-Py @("build_external_sets.py") | Out-Null

Write-Host ""
Write-Host "=== step 2: train GISR-Net ===" -ForegroundColor Cyan
Run-Config "gisrnet_tl" @("--arch", "gisrnet")

Write-Host ""
Write-Host "=== step 3: ablation study ===" -ForegroundColor Cyan
Run-Config "gisrnet_ri"      @("--arch", "gisrnet", "--no-pretrained")
Run-Config "gisrnet_noaug"   @("--arch", "gisrnet", "--no-augment")
Run-Config "gisrnet_mlp"     @("--arch", "gisrnet", "--head", "mlp")
Run-Config "gisrnet_sigmoid" @("--arch", "gisrnet", "--head", "sigmoid")

Write-Host ""
Write-Host "=== step 4: generalisability on the external sets ===" -ForegroundColor Cyan
Invoke-Py @("evaluate_external.py", "--tags", "gisrnet_tl")

Write-Host ""
Write-Host "=== step 5: figures, tables and documents ===" -ForegroundColor Cyan
Invoke-Py @("make_figures.py")
Invoke-Py @("make_tables.py")
Invoke-Py @("make_manuscript.py")

Write-Host ""
Write-Host "ALL RUNS COMPLETE" -ForegroundColor Green
Write-Host "Results are in: $RES"
