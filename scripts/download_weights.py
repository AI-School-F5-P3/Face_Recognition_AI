import os
import requests
import hashlib
from tqdm import tqdm
import argparse
from pathlib import Path

MODELS = {
    'facial_landmarks': {
        'url': 'https://github.com/opencv/opencv_3rdparty/raw/contrib/face_landmark_model.dat',
        'filename': 'facial_landmarks.dat',
        'md5': 'a587397627de68e2d3d977649018c048'  # Example MD5, replace with actual
    },
    'face_embedding': {
        'url': 'https://example.com/face_embedding_model.pth',  # Replace with actual URL
        'filename': 'face_embedding.pth',
        'md5': 'b1234567890abcdef1234567890abcde'  # Replace with actual MD5
    }
}

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url: str, filename: str, expected_md5: str = None) -> bool:
    """
    Download file from URL with progress bar and MD5 verification.
    
    Args:
        url: URL to download from
        filename: Output filename
        expected_md5: Expected MD5 hash for verification
        
    Returns:
        bool: True if download successful and MD5 matches (if provided)
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size for progress bar
        file_size = int(response.headers.get('content-length', 0))
        
        # Setup progress bar
        progress = tqdm(
            total=file_size,
            unit='iB',
            unit_scale=True,
            desc=f'Downloading {filename}'
        )
        
        # Download with progress updates
        with open(filename, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress.update(size)
        progress.close()
        
        # Verify MD5 if provided
        if expected_md5:
            actual_md5 = calculate_md5(filename)
            if actual_md5 != expected_md5:
                print(f"MD5 verification failed for {filename}")
                print(f"Expected: {expected_md5}")
                print(f"Got: {actual_md5}")
                return False
                
        return True
        
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download model weights')
    parser.add_argument('--output-dir', type=str, default='models',
                      help='Output directory for downloaded weights')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download each model
    for model_name, model_info in MODELS.items():
        print(f"\nDownloading {model_name} model...")
        
        output_path = os.path.join(args.output_dir, model_info['filename'])
        
        # Skip if file exists and MD5 matches
        if os.path.exists(output_path):
            if model_info['md5'] and calculate_md5(output_path) == model_info['md5']:
                print(f"{model_name} model already exists and MD5 matches. Skipping.")
                continue
                
        # Download model
        success = download_file(
            model_info['url'],
            output_path,
            model_info['md5']
        )
        
        if success:
            print(f"{model_name} model downloaded successfully")
        else:
            print(f"Failed to download {model_name} model")

if __name__ == '__main__':
    main()