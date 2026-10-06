# Resmi benchmark'i korur: commit bos payi 3 GB'in altina inerse uzun egitimi (u1b) durdurur.
$log = "C:\d_surucusu\visual_navigation\benchmark\deneyler\2026-09-25_parametre_arama\kayitlar\bellek_bekcisi.log"
while ($true) {
    $m = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $free = ($m.CommitLimit - $m.CommittedBytes) / 1GB
    $u1 = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'u1b_tpu_f48k4_b16' }
    if (-not $u1) { "{0} u1b yok; bekci cikiyor" -f (Get-Date -Format s) | Out-File -Append -Encoding utf8 $log; break }
    if ($free -lt 3) {
        $u1 | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
        "{0} bos commit {1:N1} GB < 3 GB -> u1b durduruldu" -f (Get-Date -Format s), $free | Out-File -Append -Encoding utf8 $log
        break
    }
    Start-Sleep -Seconds 60
}
