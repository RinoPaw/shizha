param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^https?://")]
    [string] $Url,

    [ValidateRange(1, 120)]
    [int] $TimeoutSeconds = 30
)

$ProgressPreference = "SilentlyContinue"
$healthUrl = "$($Url.TrimEnd('/'))/healthz"
$deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)

while ([DateTime]::UtcNow -lt $deadline) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 1
        if ($response.StatusCode -eq 200) {
            Start-Process -FilePath $Url
            exit 0
        }
    }
    catch {
        # The service is still starting. Retry until the deadline.
    }

    Start-Sleep -Milliseconds 250
}

exit 1
