param(
  [string]$SourcePath = "D:\VS Code\cnn-explainer",
  [string]$DestinationPath = "",
  [string]$PublicBasePath = "/mode-c/cnn-explainer"
)

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Read-Utf8Text {
  param([string]$Path)

  return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Write-Utf8Text {
  param(
    [string]$Path,
    [string]$Content
  )

  [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

$repoRoot = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($DestinationPath)) {
  $DestinationPath = Join-Path $repoRoot "frontend\public\mode-c\cnn-explainer"
}

$resolvedSourcePath = (Resolve-Path -LiteralPath $SourcePath -ErrorAction Stop).Path
$sourcePublic = Join-Path $resolvedSourcePath "public"
$requiredFiles = @(
  (Join-Path $sourcePublic "index.html"),
  (Join-Path $sourcePublic "bundle.js"),
  (Join-Path $sourcePublic "bundle.css"),
  (Join-Path $sourcePublic "global.css")
)

if (-not (Test-Path $sourcePublic)) {
  throw "Source public directory not found: $sourcePublic"
}

$missingFiles = $requiredFiles | Where-Object { -not (Test-Path $_) }
if ($missingFiles.Count -gt 0) {
  $missingList = $missingFiles -join ", "
  throw "Missing built artifacts: $missingList. Run 'npm install' and 'npm run build' in $SourcePath first."
}

$sourceAssets = Join-Path $sourcePublic "assets"
if (-not (Test-Path $sourceAssets)) {
  throw "Source assets directory not found: $sourceAssets"
}

$tfjsLocalFile = Join-Path $resolvedSourcePath "node_modules\@tensorflow\tfjs\dist\tf.min.js"
if (-not (Test-Path $tfjsLocalFile)) {
  throw "Local TensorFlow.js runtime not found: $tfjsLocalFile"
}

if (-not (Test-Path $DestinationPath)) {
  New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
}

$resolvedRepoRoot = (Resolve-Path -LiteralPath $repoRoot).Path
$resolvedDestinationPath = (Resolve-Path -LiteralPath $DestinationPath).Path
if (-not $resolvedDestinationPath.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
  throw "Refusing to sync outside repo root. Destination: $resolvedDestinationPath"
}

Copy-Item -Path (Join-Path $sourcePublic "*") -Destination $DestinationPath -Recurse -Force

$vendorDir = Join-Path $DestinationPath "vendor"
if (-not (Test-Path $vendorDir)) {
  New-Item -ItemType Directory -Path $vendorDir -Force | Out-Null
}
Copy-Item -LiteralPath $tfjsLocalFile -Destination (Join-Path $vendorDir "tf.min.js") -Force

$indexFile = Join-Path $DestinationPath "index.html"
$bundleFile = Join-Path $DestinationPath "bundle.js"
$bundleCssFile = Join-Path $DestinationPath "bundle.css"
$publicBasePathTrimmed = $PublicBasePath.TrimEnd('/')

if (Test-Path $indexFile) {
  $indexContent = Read-Utf8Text -Path $indexFile
  $indexContent = $indexContent.Replace('href="/assets/', "href=""$publicBasePathTrimmed/assets/")
  $indexContent = $indexContent.Replace('href="/global.css"', "href=""$publicBasePathTrimmed/global.css""")
  $indexContent = $indexContent.Replace('href="/bundle.css"', "href=""$publicBasePathTrimmed/bundle.css""")
  $indexContent = $indexContent.Replace('src="/bundle.js"', "src=""$publicBasePathTrimmed/bundle.js""")
  $indexContent = $indexContent.Replace("URL('/assets/", "URL('$publicBasePathTrimmed/assets/")
  $indexContent = $indexContent.Replace(
    '<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>',
    "<script src=""$publicBasePathTrimmed/vendor/tf.min.js""></script>"
  )
  $bootScript = @"
    <script>
      (function () {
        var originalInsertBefore = Node.prototype.insertBefore;
        Node.prototype.insertBefore = function (newNode, referenceNode) {
          if (
            newNode &&
            newNode.tagName === 'SCRIPT' &&
            typeof newNode.src === 'string' &&
            newNode.src.indexOf('youtube.com/iframe_api') !== -1
          ) {
            return newNode;
          }
          return originalInsertBefore.call(this, newNode, referenceNode);
        };

        function showBootError(message) {
          var existing = document.getElementById('cnn-explainer-boot-error');
          if (existing) {
            existing.textContent = message;
            return;
          }
          var box = document.createElement('pre');
          box.id = 'cnn-explainer-boot-error';
          box.textContent = message;
          box.style.cssText = 'position:fixed;top:12px;left:12px;right:12px;z-index:99999;padding:12px 14px;border-radius:12px;background:#fff1f2;color:#9f1239;border:1px solid #fecdd3;font:12px/1.6 Consolas,monospace;white-space:pre-wrap;';
          document.body.appendChild(box);
        }

        window.addEventListener('error', function (event) {
          var target = event && event.target;
          if (target && target.tagName === 'SCRIPT' && target.src) {
            if (target.src.indexOf('youtube.com/iframe_api') !== -1) {
              return;
            }
            showBootError('CNN Explainer failed to load script:\n' + target.src);
            return;
          }
          showBootError('CNN Explainer runtime error:\n' + (event.message || 'Unknown error'));
        }, true);

        window.addEventListener('unhandledrejection', function (event) {
          var reason = event && event.reason;
          showBootError('CNN Explainer promise rejection:\n' + (reason && reason.message ? reason.message : String(reason)));
        });

        function removeVideoTutorialSection() {
          var description = document.getElementById('description');
          if (!description) {
            return false;
          }

          var headings = Array.prototype.slice.call(description.querySelectorAll('h2'));
          var tutorialHeading = headings.find(function (heading) {
            return (heading.textContent || '').trim() === 'Video Tutorial';
          });

          if (!tutorialHeading) {
            return true;
          }

          var nodesToRemove = [];
          var current = tutorialHeading;
          while (current) {
            if (
              current !== tutorialHeading &&
              current.tagName === 'H2'
            ) {
              break;
            }
            nodesToRemove.push(current);
            current = current.nextElementSibling;
          }

          nodesToRemove.forEach(function (node) {
            node.remove();
          });

          return true;
        }

        window.setTimeout(function () {
          if (!document.getElementById('app-page')) {
            var d3Ready = typeof window.d3 !== 'undefined';
            var tfReady = typeof window.tf !== 'undefined';
            showBootError(
              'CNN Explainer did not finish booting.\n' +
              'd3 loaded: ' + d3Ready + '\n' +
              'tf loaded: ' + tfReady + '\n' +
              'If d3 is false, the external CDN script is likely blocked.'
            );
            return;
          }

          removeVideoTutorialSection();
        }, 5000);

        var cleanupTimer = window.setInterval(function () {
          if (removeVideoTutorialSection()) {
            window.clearInterval(cleanupTimer);
          }
        }, 500);
      })();
    </script>
"@
  $indexContent = $indexContent.Replace('</head>', "$bootScript`r`n  </head>")
  Write-Utf8Text -Path $indexFile -Content $indexContent
}

if (Test-Path $bundleFile) {
  $bundleContent = Read-Utf8Text -Path $bundleFile
  $bundleContent = $bundleContent.Replace('/cnn-explainer/assets/', "$publicBasePathTrimmed/assets/")
  Write-Utf8Text -Path $bundleFile -Content $bundleContent
}

if (Test-Path $bundleCssFile) {
  $bundleCssContent = Read-Utf8Text -Path $bundleCssFile
  $bundleCssContent = $bundleCssContent.Replace('/cnn-explainer/assets/', "$publicBasePathTrimmed/assets/")
  Write-Utf8Text -Path $bundleCssFile -Content $bundleCssContent
}

Write-Host "cnn-explainer static bundle synced to $resolvedDestinationPath"
