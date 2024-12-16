import pytest
import requests
from pathlib import Path
import gzip
import shutil


@pytest.fixture(scope="session", autouse=True)
def ensure_downloaded_and_extracted():
    urls = {
        "gen-ip054.mps.gz": "https://miplib.zib.de/WebData/instances/gen-ip054.mps.gz",
        "flugpl.mps.gz": "https://miplib.zib.de/WebData/instances/flugpl.mps.gz",
    }
    project_dir = Path(__file__).parent.parent
    cache_dir = project_dir / ".pytest_cache"
    cache_dir.mkdir(exist_ok=True)

    for filename, url in urls.items():
        gz_file_path = cache_dir / filename
        print(f"Checking '{gz_file_path}'...")
        extracted_file_path = cache_dir / filename.replace(".gz", "")

        if not gz_file_path.exists():
            print(f"Downloading '{filename}'...")
            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(gz_file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                print(f"File '{filename}' downloaded successfully.")
            except requests.exceptions.RequestException as e:
                pytest.exit(f"Failed to download '{filename}': {e}")
        else:
            print(f"File '{filename}' already exists. Skipping download.")

        if not extracted_file_path.exists():
            print(f"Extracting '{filename}'...")
            try:
                with gzip.open(gz_file_path, "rb") as gz_file:
                    with open(extracted_file_path, "wb") as extracted_file:
                        shutil.copyfileobj(gz_file, extracted_file)
                print(f"File '{extracted_file_path.name}' extracted successfully.")
            except Exception as e:
                pytest.exit(f"Failed to extract '{filename}': {e}")
        else:
            print(
                f"File '{extracted_file_path.name}' already extracted. Skipping extraction."
            )

    print("All required files are ready.")
    return cache_dir
