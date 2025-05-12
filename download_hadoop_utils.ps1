$hadoopBinUrl = "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.0/bin/winutils.exe"
$hadoopDllUrl = "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.0/bin/hadoop.dll"
$hdfsUrl = "https://github.com/kontext-tech/winutils/raw/master/hadoop-3.3.0/bin/hdfs.dll"

# Create hadoop/bin directory if it doesn't exist
$binPath = "hadoop\bin"
New-Item -ItemType Directory -Force -Path $binPath

# Download files using WebClient
Write-Host "Downloading Hadoop utilities..."
$webClient = New-Object System.Net.WebClient

try {
    $webClient.DownloadFile($hadoopBinUrl, "$binPath\winutils.exe")
    Write-Host "Downloaded winutils.exe"
    $webClient.DownloadFile($hadoopDllUrl, "$binPath\hadoop.dll")
    Write-Host "Downloaded hadoop.dll"
    $webClient.DownloadFile($hdfsUrl, "$binPath\hdfs.dll")
    Write-Host "Downloaded hdfs.dll"
    Write-Host "Download complete. Files saved to $binPath"
} catch {
    Write-Host "Error downloading files: $_"
} finally {
    $webClient.Dispose()
} 