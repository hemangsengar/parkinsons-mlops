import joblib
import yaml

class ParkinsonModel:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = joblib.load(self.config["model_path"])

    def predict(self, df):
        return self.model.predict(df)[0]
