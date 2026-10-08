$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime] | Out-Null
$taskMethod = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await-Operation($Operation, [Type]$ResultType) {
    $task = $taskMethod.MakeGenericMethod($ResultType).Invoke($null,@($Operation))
    $task.Wait()
    return $task.Result
}
$engine=[Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { throw 'English Windows OCR engine unavailable' }
$root=Split-Path -Parent $PSScriptRoot
$dest=Join-Path $root 'outputs\cv_knowledge_base\extracted\ocr'
New-Item -ItemType Directory -Path $dest -Force | Out-Null
$jobs=Get-Content -LiteralPath (Join-Path $PSScriptRoot 'ocr_jobs.json') -Raw | ConvertFrom-Json
foreach ($job in $jobs) {
    $out=Join-Path $dest ($job.id+'.json')
    if (Test-Path -LiteralPath $out) { continue }
    $file=Await-Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($job.path)) ([Windows.Storage.StorageFile])
    $stream=Await-Operation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder=Await-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap=Await-Operation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result=Await-Operation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $lines=@()
    foreach ($line in $result.Lines) {
        $words=@()
        foreach ($word in $line.Words) {
            $rect=$word.BoundingRect
            $words+=@{text=$word.Text;x=$rect.X;y=$rect.Y;width=$rect.Width;height=$rect.Height}
        }
        $lines+=@{text=$line.Text;words=$words}
    }
    @{id=$job.id;source_id=$job.source_id;page=$job.page;text=$result.Text;lines=$lines;engine='Windows.Media.Ocr en-US';review_status='raw_ocr_unverified'} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $out -Encoding UTF8
    $bitmap.Dispose(); $stream.Dispose()
    Write-Output $job.id
}
