
from src.model import ParkinsonModel
from src.preprocess import preprocess_input

model = ParkinsonModel()

def predict_parkinson(input_dict):
    features = preprocess_input(input_dict)
    pred = model.predict(features)
    return "Parkinson's" if pred == 1 else "Healthy"
