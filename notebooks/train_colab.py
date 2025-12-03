!pip install pandas numpy scikit-learn joblib
from google.colab import files
uploaded = files.upload()          #you can download parkinson's dataset from KAGGLE- https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set
os.rename("parkinsons (1).data", "parkinsons.csv")
import pandas as pd
df = pd.read_csv("parkinsons.csv")
df.head()
df = df.drop("name", axis=1)
X = df.drop("status", axis=1)
y = df["status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
pred = model.predict(X_test)
accuracy_score(y_test, pred)
import joblib
joblib.dump(model, "model.pkl")



