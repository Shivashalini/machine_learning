
import pickle
import pandas as pd
from flask import Flask

app = Flask(__name__)

# Corrected DataFrame
df = pd.DataFrame(
    [[56, 'female', 19.950, 2, 'no', 'northwest']],
    columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']
)

# Load pipeline
with open(r"D:\data science ppt\trainig2.pkl", "rb") as file:
    pipeline = pickle.load(file)

# Prediction
print(pipeline.predict(df))

print("Prediction:", prediction)

