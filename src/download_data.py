import os
import requests
import zipfile
import io

# List of URLs extracted from the HTML
urls = [
    "https://database.nikonoel.fr/lichess_elite_2020-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2020-12.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-01.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-02.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-03.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-04.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-05.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2021-12.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-01.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-02.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-03.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-04.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-05.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2022-12.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-01.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-02.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-03.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-04.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-05.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2023-12.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-01.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-02.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-03.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-04.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-05.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2024-12.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-01.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-02.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-03.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-04.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-05.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-06.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-11.zip"
]

# Directory to save the PGN files
download_dir = "lichess_elite_db"

if not os.path.exists(download_dir):
    os.makedirs(download_dir)

print(f"Starting download of {len(urls)} files to '{download_dir}'...")

for url in urls:
    try:
        filename = url.split("/")[-1]
        print(f"Downloading {filename}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Use io.BytesIO to handle the zip file in memory
        z = zipfile.ZipFile(io.BytesIO(response.content))
        
        # Extract all contents to the download directory
        z.extractall(download_dir)
        print(f"Successfully extracted {filename}")
        
    except Exception as e:
        print(f"Failed to download or extract {url}: {e}")

print("All tasks completed.")