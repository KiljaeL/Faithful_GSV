import os
import requests
import tarfile
import shutil

url = "http://image.ntua.gr/iva/datasets/flickr_logos/flickr_logos_27_dataset.tar.gz"

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
output_path = os.path.join(output_dir, "flickr_logos_27_dataset.tar.gz")
temp_extract_dir = os.path.join(output_dir, "flickr_logos_27_dataset")

os.makedirs(output_dir, exist_ok=True)

response = requests.get(url, stream=True)
with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print("Download completed:", output_path)

with tarfile.open(output_path, "r:gz") as tar:
    tar.extractall(path=output_dir)


for item in os.listdir(temp_extract_dir):
    src_path = os.path.join(temp_extract_dir, item)
    dst_path = os.path.join(output_dir, item)
    if os.path.exists(dst_path):
        print(f"Warning: {dst_path} already exists and will be overwritten.")
    if os.path.isdir(src_path):
        shutil.move(src_path, dst_path)
    else:
        shutil.move(src_path, output_dir)

shutil.rmtree(temp_extract_dir)

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
archive_path = os.path.join(data_dir, "flickr_logos_27_dataset_images.tar.gz")

with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=data_dir)

print("Extraction completed.")
