param(
    [string]$OutputRoot = "Review/output",
    [int]$MinYear = 2019,
    [int]$MaxYear = 2026,
    [int]$MaxResultsPerQuery = 300,

    [string]$ScopusLanguage = "english",
    [string]$ScopusDoctypes = "ar",
    [int]$ScopusPageSize = 25,
    [ValidateSet("STANDARD", "COMPLETE")]
    [string]$ScopusView = "COMPLETE",

    [string]$WosLanguage = "English",
    [string]$WosDocTypes = "Article",
    [int]$WosPageSize = 50,
    [ValidateSet("full", "short")]
    [string]$WosDetail = "full",
    [string]$WosDb = "WOS",
    [ValidateSet("expanded", "starter")]
    [string]$WosApiMode = "expanded",
    [ValidateSet("SR", "FR")]
    [string]$WosOptionView = "SR",
    [string]$WosExpandedEndpoint = "https://wos-api.clarivate.com/api/wos",

    [switch]$SkipScopus,
    [switch]$SkipWos
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-PythonScript {
    param(
        [string[]]$ArgsList
    )
    & python @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "Python script failed (exit code: $LASTEXITCODE): python $($ArgsList -join ' ')"
    }
}

function New-Dir {
    param([string]$PathValue)
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Get-YearFromRow {
    param($Row)
    if ($null -ne $Row.PSObject.Properties["publish_year"] -and $Row.publish_year) {
        return "$($Row.publish_year)".Trim()
    }
    if ($null -ne $Row.PSObject.Properties["cover_date"] -and $Row.cover_date) {
        $cd = "$($Row.cover_date)".Trim()
        if ($cd.Length -ge 4) {
            return $cd.Substring(0, 4)
        }
    }
    return ""
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $repoRoot

$scopusScript = Join-Path $scriptDir "scopus_advanced_search.py"
$wosScript = Join-Path $scriptDir "wos_advanced_search.py"

$scopusOut = Join-Path $OutputRoot "scopus"
$wosOut = Join-Path $OutputRoot "wos"
$consolidatedOut = Join-Path $OutputRoot "consolidated"

New-Dir -PathValue $OutputRoot
New-Dir -PathValue $scopusOut
New-Dir -PathValue $wosOut
New-Dir -PathValue $consolidatedOut

$ranScopus = $false
$ranWos = $false
$errors = @()

if (-not $SkipScopus) {
    if ([string]::IsNullOrWhiteSpace($env:ELS_API_KEY)) {
        Write-Warning "ELS_API_KEY is not set. Skipping Scopus run."
    }
    else {
        try {
            Write-Host "[run] Scopus"
            Invoke-PythonScript -ArgsList @(
                $scopusScript,
                "--output-dir", $scopusOut,
                "--min-year", "$MinYear",
                "--language", $ScopusLanguage,
                "--doctypes", $ScopusDoctypes,
                "--max-results-per-query", "$MaxResultsPerQuery",
                "--page-size", "$ScopusPageSize",
                "--view", $ScopusView
            )
            $ranScopus = $true
        }
        catch {
            $msg = "Scopus run failed: $($_.Exception.Message)"
            $errors += $msg
            Write-Warning $msg
        }
    }
}

if (-not $SkipWos) {
    if ([string]::IsNullOrWhiteSpace($env:WOS_API_KEY)) {
        Write-Warning "WOS_API_KEY is not set. Skipping Web of Science run."
    }
    else {
        try {
            Write-Host "[run] Web of Science"
            Invoke-PythonScript -ArgsList @(
                $wosScript,
                "--output-dir", $wosOut,
                "--api-mode", $WosApiMode,
                "--expanded-endpoint", $WosExpandedEndpoint,
                "--db", $WosDb,
                "--min-year", "$MinYear",
                "--max-year", "$MaxYear",
                "--language", $WosLanguage,
                "--doc-types", $WosDocTypes,
                "--max-results-per-query", "$MaxResultsPerQuery",
                "--page-size", "$WosPageSize",
                "--detail", $WosDetail,
                "--option-view", $WosOptionView
            )
            $ranWos = $true
        }
        catch {
            $msg = "Web of Science run failed: $($_.Exception.Message)"
            $errors += $msg
            Write-Warning $msg
        }
    }
}

$scopusDedup = Join-Path $scopusOut "scopus_all_queries_dedup.csv"
$wosDedup = Join-Path $wosOut "wos_all_queries_dedup.csv"

$combinedRows = @()

if (Test-Path $scopusDedup) {
    $rows = Import-Csv $scopusDedup
    foreach ($r in $rows) {
        $obj = [ordered]@{ source_db = "Scopus" }
        foreach ($p in $r.PSObject.Properties) {
            $obj[$p.Name] = $p.Value
        }
        $combinedRows += [pscustomobject]$obj
    }
}

if (Test-Path $wosDedup) {
    $rows = Import-Csv $wosDedup
    foreach ($r in $rows) {
        $obj = [ordered]@{ source_db = "WebOfScience" }
        foreach ($p in $r.PSObject.Properties) {
            $obj[$p.Name] = $p.Value
        }
        $combinedRows += [pscustomobject]$obj
    }
}

$combinedRawPath = Join-Path $consolidatedOut "combined_crossdb_raw.csv"
$combinedDedupPath = Join-Path $consolidatedOut "combined_crossdb_dedup.csv"
$reportPath = Join-Path $consolidatedOut "run_report.json"

if ($combinedRows.Count -gt 0) {
    $combinedRows | Export-Csv -Path $combinedRawPath -NoTypeInformation -Encoding UTF8

    $seen = @{}
    $dedup = @()

    foreach ($row in $combinedRows) {
        $doi = ""
        if ($null -ne $row.PSObject.Properties["doi"] -and $row.doi) {
            $doi = "$($row.doi)".Trim().ToLowerInvariant()
        }

        $key = ""
        if (-not [string]::IsNullOrWhiteSpace($doi)) {
            $key = "doi:$doi"
        }
        else {
            $title = ""
            if ($null -ne $row.PSObject.Properties["title"] -and $row.title) {
                $title = ("$($row.title)".Trim().ToLowerInvariant() -replace "\s+", " ")
            }
            $year = Get-YearFromRow -Row $row
            $key = "ty:$title|$year"
        }

        if (-not $seen.ContainsKey($key)) {
            $seen[$key] = $true
            $dedup += $row
        }
    }

    $dedup | Export-Csv -Path $combinedDedupPath -NoTypeInformation -Encoding UTF8
}

$report = [ordered]@{
    timestamp = (Get-Date).ToString("s")
    ran_scopus = $ranScopus
    ran_wos = $ranWos
    scopus_dedup_exists = (Test-Path $scopusDedup)
    wos_dedup_exists = (Test-Path $wosDedup)
    combined_raw_exists = (Test-Path $combinedRawPath)
    combined_dedup_exists = (Test-Path $combinedDedupPath)
    output_root = (Resolve-Path $OutputRoot).Path
    errors = $errors
}

$report | ConvertTo-Json -Depth 5 | Out-File -FilePath $reportPath -Encoding utf8

Write-Host "[done] Report: $reportPath"
if (Test-Path $combinedRawPath) {
    Write-Host "[done] Combined raw: $combinedRawPath"
}
if (Test-Path $combinedDedupPath) {
    Write-Host "[done] Combined dedup: $combinedDedupPath"
}
