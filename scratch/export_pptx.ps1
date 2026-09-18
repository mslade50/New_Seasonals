param(
  [string]$Pptx = "C:\Users\McKinley Slade\dev\New_Seasonals\presentation\The_Book.pptx",
  [string]$OutDir = "C:\Users\McKinley Slade\dev\New_Seasonals\scratch\pptx_shots"
)
$ErrorActionPreference = "Stop"
if (Test-Path $OutDir) { Remove-Item "$OutDir\*" -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$pp = New-Object -ComObject PowerPoint.Application
try {
  $pres = $pp.Presentations.Open($Pptx, $true, $false, $false)
  $pres.Export($OutDir, "PNG", 1600, 900)
  $pres.Close()
  Write-Output "EXPORTED to $OutDir"
} finally {
  $pp.Quit()
  [System.Runtime.Interopservices.Marshal]::ReleaseComObject($pp) | Out-Null
}
