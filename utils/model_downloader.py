import os
import gdown
import torch
from pathlib import Path


class ModelDownloader:
    def __init__(self):
        self.file_id = "1dTVWPJOLATaLTfIMrZHNbUEAvDszfpfx"
        self.model_name = "AffectNet7_Model.pth"
        
        # get the project root directory
        self.project_root = Path(__file__).parent.parent
        self.model_dir = self.project_root / "trained_model"
        self.model_path = self.model_dir / self.model_name


    def download_model(self):
        """
        Downloads the model file from Google Drive if it doesn't exist locally.

        Returns:
            str: Path to the downloaded model file

        Raises:
            Exception: If the model file cannot be downloaded or found
        """

        # create directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        if not os.path.exists(self.model_path):
            print("Model file not found locally.")
            print("Downloading model file from Google Drive...")
            try:
                # URL format for Google Drive
                url = f"https://drive.google.com/uc?id={self.file_id}"
                gdown.download(url, str(self.model_path), quiet=False)
                print(f"Model downloaded successfully to {self.model_path}")
            except Exception as e:
                print(f"Error downloading model: {str(e)}")
                raise Exception("Failed to download model file. Please check your internet connection or download the file manually.")
        else:
            print("Model file already exists locally.")
        
        return str(self.model_path)


    def verify_model(self, model_path):
        """
        Verify if the model file is valid and can be loaded.

        Args:
            model_path (str): Path to the model file
        
        Raises:
            Exception: If the model file is invalid or cannot be loaded
        """
        try:
            torch.load(model_path, map_location='cpu', weights_only=True)
            return True
        except Exception as e:
            print(f"Warning: Model verification failed: {str(e)}")
            try:
                os.remove(model_path)
                print(f"Removed invalid model file: {model_path}")
            except Exception:
                pass
            return False
          

    def verify_model_exists(self) -> bool:
        """
        Checks if the model file exists locally.

        Returns:
            bool: True if the model file exists, False otherwise
        """
        return os.path.exists(self.model_path) and self.verify_model(self.model_path)


    def get_model_path(self) -> str:
        """
        Returns the path to the model file.

        Returns:
            str: Path to the model file
        """
        return str(self.model_path) 
    