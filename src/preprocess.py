import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_input(data: dict, feature_list: list):
    df = pd.DataFrame([data])[feature_list]
    return df
