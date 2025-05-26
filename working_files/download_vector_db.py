import subprocess
import sys
import os
from datetime import datetime, timedelta

def run_command(cmd, capture_output):
    """Run a shell command and return output."""
    try:
        
        shell_mode = os.name == 'nt'
        if capture_output:
            result = subprocess.run(cmd, shell=shell_mode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=shell_mode, check=True)
            return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e.stderr}")
        sys.exit(1)

def main():
    # User-configurable variables
    account_name = "taicdocumentsearcherdata"
    container_name = "vectordb"
    blob_path = "prod/all_document_types.lance"
    local_dir = "./testing"

    # Set expiry to 1 day from now in UTC
    expiry = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"Using SAS expiry time: {expiry}")

    # Generate SAS token command
    sas_cmd = [
        "az", "storage", "container", "generate-sas",
        "--account-name", account_name,
        "--name", container_name,
        "--permissions", "rl",
        "--expiry", expiry,
        "--output", "tsv",
    ]

    print("Generating SAS token...")
    sas_token = run_command(sas_cmd, capture_output=True)
    sas_token = sas_token.strip()

    if blob_path.startswith('/'):
        blob_path = blob_path[1:]

    blob_url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}?{sas_token}"

    print(f"Blob URL with SAS token:\n{blob_url}")

    azcopy_cmd = [
        "azcopy", "copy",
        blob_url,
        local_dir,
        "--recursive"
    ]

    print("Starting azcopy download...")
    run_command(azcopy_cmd, capture_output=False)
    print("Download completed.")

if __name__ == "__main__":
    main()
