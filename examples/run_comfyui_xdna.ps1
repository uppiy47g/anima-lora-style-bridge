param(
    [Parameter(Mandatory = $true)]
    [string]$ComfyUIRoot,
    [Parameter(Mandatory = $true)]
    [string]$XDNAVenv,
    [Parameter(Mandatory = $true)]
    [string]$XRTDevDir,
    [Parameter(Mandatory = $true)]
    [string]$ModelPathsConfig,
    [string]$InputDirectory = "",
    [string]$OutputDirectory = "",
    [int]$Port = 8190,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ComfyUIArgs
)

$ErrorActionPreference = "Stop"
$python = Join-Path $XDNAVenv "Scripts\python.exe"
$main = Join-Path $ComfyUIRoot "main.py"
foreach ($path in @($python, $main, $XRTDevDir, $ModelPathsConfig)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$env:XRT_DEV_DIR = (Resolve-Path -LiteralPath $XRTDevDir).Path
$env:PYTHONUTF8 = "1"
$arguments = @(
    $main,
    "--cpu",
    "--listen", "127.0.0.1",
    "--port", $Port,
    "--extra-model-paths-config", (Resolve-Path -LiteralPath $ModelPathsConfig).Path
)
if ($InputDirectory) {
    $arguments += @("--input-directory", $InputDirectory)
}
if ($OutputDirectory) {
    $arguments += @("--output-directory", $OutputDirectory)
}
$arguments += $ComfyUIArgs

& $python @arguments
exit $LASTEXITCODE
