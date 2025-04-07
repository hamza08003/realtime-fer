import os
import sys
import torch
from pathlib import Path
import torch.nn.functional as F
from utils.model_downloader import ModelDownloader
from nn_architecture.ResEmote_Net import ResEmoteNet


# adding the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


class EmotionPredictor:
    def __init__(self, checkpoint_path, device):
        self.device = device
        model_downloader = ModelDownloader()

        try:
            checkpoint_path = model_downloader.download_model()
        except Exception as e:
            print(f"Error ensuring model availability: {str(e)}")
            raise
        
        self.model = ResEmoteNet().to(device)

        try:
            torch.serialization.add_safe_globals({"ResEmoteNet": ResEmoteNet})
            checkpoint = torch.load(
                checkpoint_path, 
                map_location=device, 
                weights_only=True
            )
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise Exception("Failed to load model. The model file might be corrupted.")
            
        self.model.eval()
        self.labels = ["neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]


    def predict(self, input_tensor):
        """
        Predict emotion from a preprocessed tensor.
        Args:
            input_tensor: Preprocessed tensor (1, 3, 64, 64)
        Returns:
            Predicted label and probabilities
        """
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            predicted_label = self.labels[pred_class.item()]
            probabilities = probs.cpu().numpy()[0] * 100
        return predicted_label, probabilities
    