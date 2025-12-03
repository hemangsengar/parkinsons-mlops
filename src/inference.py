from src.model import ParkinsonModel
from src.preprocess import preprocess_input
import yaml

def run_inference(input_data: dict):
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    features = config["features"]

    df = preprocess_input(input_data, features)
    model = ParkinsonModel()
    prediction = model.predict(df)

    return prediction
