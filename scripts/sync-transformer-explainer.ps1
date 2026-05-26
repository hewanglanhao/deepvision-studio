param(
  [string]$SourceRoot = "D:\VS Code\transformer-explainer",
  [string]$FrontendRoot = "D:\VS Code\deep-learning-plat-form\frontend",
  [string]$TargetName = "mode-d-assets"
)

$ErrorActionPreference = "Stop"

$sourceStatic = Join-Path $SourceRoot "static"
$targetRoot = Join-Path $FrontendRoot "public\$TargetName"
$vendorRoot = Join-Path $targetRoot "vendor\onnxruntime"
$transformersVendorRoot = Join-Path $targetRoot "vendor\transformers"
$onnxRuntimeDistRoot = Join-Path $FrontendRoot "node_modules\onnxruntime-web\dist"
$transformersDistRoot = Join-Path $FrontendRoot "node_modules\@xenova\transformers\dist"

if (-not (Test-Path $sourceStatic)) {
  throw "未找到 transformer-explainer 静态资源目录：$sourceStatic"
}

$resolvedPublicRoot = (Resolve-Path (Join-Path $FrontendRoot "public")).Path
if (-not ($resolvedPublicRoot -like "D:\VS Code\deep-learning-plat-form\frontend\public*")) {
  throw "目标目录不在当前仓库允许范围内：$resolvedPublicRoot"
}

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$syncDirs = @("model-v2", "article_assets", "preview")
foreach ($dir in $syncDirs) {
  $sourceDir = Join-Path $sourceStatic $dir
  $targetDir = Join-Path $targetRoot $dir

  if (-not (Test-Path $sourceDir)) {
    throw "缺少源目录：$sourceDir"
  }

  New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
  Copy-Item -Path (Join-Path $sourceDir "*") -Destination $targetDir -Recurse -Force
}

if (Test-Path $onnxRuntimeDistRoot) {
  New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null

  Get-ChildItem -Path $onnxRuntimeDistRoot -Filter "ort-wasm*.wasm" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $vendorRoot -Force
  }

  Get-ChildItem -Path $onnxRuntimeDistRoot -Filter "ort-wasm*.mjs" | ForEach-Object {
    Copy-Item -Path $_.FullName -Destination $vendorRoot -Force
    $jsTarget = Join-Path $vendorRoot ($_.BaseName + ".js")
    Copy-Item -Path $_.FullName -Destination $jsTarget -Force
  }
}
else {
  Write-Warning "未找到 onnxruntime-web dist 目录，已跳过 wasm/mjs 文件同步：$onnxRuntimeDistRoot"
}

if (Test-Path $transformersDistRoot) {
  New-Item -ItemType Directory -Force -Path $transformersVendorRoot | Out-Null
  Copy-Item -Path (Join-Path $transformersDistRoot "transformers.min.js") -Destination $transformersVendorRoot -Force
}
else {
  Write-Warning "未找到 transformers 浏览器分发目录，已跳过脚本同步：$transformersDistRoot"
}

Write-Host "Transformer Explainer 资源已同步到 $targetRoot"
