import kagglehub
import shutil
import os

# Download dataset using KaggleHub
path = kagglehub.dataset_download("adityamhaske/bank-marketing-dataset")

print("Downloaded to:", path)

# Destination folder
destination = "datasets"

# Create destination folder if it doesn't exist
os.makedirs(destination, exist_ok=True)

# Copy all downloaded files to your destination
for file_name in os.listdir(path):
    source = os.path.join(path, file_name)
    destination_file = os.path.join(destination, file_name)

    if os.path.isfile(source):
        shutil.copy2(source, destination_file)

print("Dataset copied to:", destination)
print("Files:", os.listdir(destination))