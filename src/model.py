import joblib

class ParkinsonModel:
    def __init__(self, model_path="models/model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, features):
        return self.model.predict([features])[0]

