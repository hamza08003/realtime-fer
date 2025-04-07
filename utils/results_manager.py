import pandas as pd
import os


class ResultsManager:
    def __init__(self, results_file="results.csv"):
        self.results_file = results_file
        self.results = []
        if os.path.exists(results_file):
            self.results = pd.read_csv(results_file).to_dict('records')

    def add_result(self, image_name, predicted_label, probabilities):
        result = {
            "Image": image_name,
            "Predicted Emotion": predicted_label,
            "Neutral": f"{probabilities[0]:.2f}%",
            "Happiness": f"{probabilities[1]:.2f}%",
            "Sadness": f"{probabilities[2]:.2f}%",
            "Surprise": f"{probabilities[3]:.2f}%",
            "Fear": f"{probabilities[4]:.2f}%",
            "Disgust": f"{probabilities[5]:.2f}%",
            "Anger": f"{probabilities[6]:.2f}%"
        }
        self.results.append(result)
        pd.DataFrame(self.results).to_csv(self.results_file, index=False)

    def get_results(self):
        return pd.DataFrame(self.results)