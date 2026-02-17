param()
$response = Invoke-WebRequest -Uri "http://localhost:5000/api/race/replay-data?year=2024&round=21" -UseBasicParsing | Select-Object -ExpandProperty Content | ConvertFrom-Json
$lap2 = @($response.frames | Where-Object {$_.lap -eq 2})

Write-Host "Lap 2 Frame 1 - Driver X,Y positions:"
$frame = $lap2[0]
$positions = @()

foreach ($code in @('VER', 'LEC', 'NOR', 'HAM', 'RUS')) {
    $driver = $frame.drivers.$code
    if ($driver) {
        $x = [math]::Round($driver.x, 0)
        $y = [math]::Round($driver.y, 0)
        $pos = $driver.position
        Write-Host ("  {0}: pos={1:F1}, x={2}, y={3}" -f $code, $pos, $x, $y)
        $positions += @{code=$code; x=$driver.x; y=$driver.y}
    }
}

# Check spread
if ($positions.Count -gt 1) {
    $xs = $positions | ForEach-Object { $_.x }
    $ys = $positions | ForEach-Object { $_.y }
    $xRange = (($xs | Measure-Object -Maximum).Maximum - ($xs | Measure-Object -Minimum).Minimum)
    $yRange = (($ys | Measure-Object -Maximum).Maximum - ($ys | Measure-Object -Minimum).Minimum)
    Write-Host "`nDriver spread across track:"
    Write-Host ("  X range: {0:F0}" -f $xRange)
    Write-Host ("  Y range: {0:F0}" -f $yRange)
    if ($xRange -gt 500 -or $yRange -gt 500) {
        Write-Host "  ✓ Good spread - drivers not clustered at finish line!"
    }
}
