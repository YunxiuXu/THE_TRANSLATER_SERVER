# 一键安装：检查显卡 -> 装 Python / Ollama -> 下载翻译模型 -> 建虚拟环境 -> 预下载 Whisper
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
}

Step '检查 NVIDIA 显卡'
try {
    $vram = [int](nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1)
    $gpu = nvidia-smi --query-gpu=name --format=csv,noheader | Select-Object -First 1
} catch {
    Write-Host '没找到 NVIDIA 显卡或驱动（nvidia-smi 不可用）。本程序需要 NVIDIA 显卡。' -ForegroundColor Red
    exit 1
}
# 12GB 以上用 12B 翻译模型；更小的显存用 4B（质量差一些）
$model = if ($vram -ge 11000) { 'translategemma:12b' } else { 'translategemma:4b' }
Write-Host "$gpu, $([math]::Round($vram / 1024))GB 显存 -> 翻译模型 $model"
Set-Content -Encoding ascii config.bat "set MODEL=$model"

Step '检查 Python'
$py = $null
foreach ($v in '3.12', '3.11', '3.10') {
    try { & py "-$v" -c 'import sys' 2>$null; if ($LASTEXITCODE -eq 0) { $py = @('py', "-$v"); break } } catch {}
}
if (-not $py) {
    Write-Host '没找到 Python 3.10-3.12，用 winget 安装 Python 3.12……'
    winget install --id Python.Python.3.12 -e --silent --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $py = @('py', '-3.12')
}

Step '检查 Ollama'
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host '没找到 Ollama，用 winget 安装……'
    winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements
    Refresh-Path
    $exe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and (Test-Path $exe)) {
        $env:Path += ";$(Split-Path $exe)"
    }
}
# 确保 Ollama 服务在跑
try { Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 3 | Out-Null }
catch { Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 5 }

Step "下载翻译模型 $model"
ollama pull $model

Step '创建 Python 虚拟环境并安装依赖'
if (-not (Test-Path .venv\Scripts\python.exe)) { & $py[0] $py[1] -m venv .venv }
.venv\Scripts\python.exe -m pip install --upgrade pip -q
.venv\Scripts\python.exe -m pip install -r requirements.txt

Step '预下载语音识别模型 Whisper large-v3-turbo（约 1.6GB）并测试显卡'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = 1
.venv\Scripts\python.exe -c "import live_translate as lt, numpy as np; m = lt.WhisperModel('large-v3-turbo', device='cuda', compute_type='float16'); list(m.transcribe(np.zeros(16000, np.float32))[0]); print('Whisper OK')"

Write-Host "`n安装完成！双击 start.bat 启动实时翻译。" -ForegroundColor Green
