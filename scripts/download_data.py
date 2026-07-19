# /// script
# dependencies = [
#   "requests"
# ]
# ///
import os
import zipfile
import requests

def download_file(url, dest):
    print(f"Downloading {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed.")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    zip_path = os.path.join(project_dir, "DevanagariDataset.zip")
    extract_dir = os.path.join(project_dir, "data")
    dataset_dir = os.path.join(extract_dir, "DevanagariHandwrittenCharacterDataset")
    
    if os.path.exists(dataset_dir):
        print(f"Dataset already exists in: {dataset_dir}")
        print("No need to download again!")
        return

    # Note: The UCI URL often 404s now because the dataset was moved.
    # The official source is now primarily Kaggle: 
    # https://www.kaggle.com/datasets/rishianand/devanagari-character-set
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00389/DevanagariHandwrittenCharacterDataset.zip"
    
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
        
    if not os.path.exists(zip_path):
        try:
            download_file(url, zip_path)
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return
            
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction completed successfully.")
        os.remove(zip_path)
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid ZIP archive.")
        print("This usually happens because the UCI download link is broken (returns a 404 webpage).")
        print("Please download the dataset manually from Kaggle:")
        print("https://www.kaggle.com/datasets/rishianand/devanagari-character-set")
        print("Extract it and place the 'DevanagariHandwrittenCharacterDataset' folder into the 'data' directory.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == '__main__':
    main()
