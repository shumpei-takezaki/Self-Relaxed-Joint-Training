import requests
import zipfile
from pathlib import Path

root_dir = Path("./dataset/limuc")
root_dir.mkdir(exist_ok=True, parents=True)
url = "https://zenodo.org/records/5827695/files/patient_based_classified_images.zip?download=1"
output_file = root_dir.joinpath("patient_based_classified_images.zip")
extrat_to = root_dir.joinpath("patient_based_classified_images")

if not output_file.exists(): 
    response = requests.get(url, stream=True)
    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

print("Download complete.")

if not extrat_to.exists():
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(root_dir)

print("Files extracts")