import joblib
import numpy as np 

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from data_preprocessing import preprocess_data


# =========================
# 1. Load & preprocess data
# =========================
DATA_PATH = "../data/raw/train.csv"

X_train, X_val, y_train, y_val, preprocessor = preprocess_data(DATA_PATH)


# =========================
# 2. Build training pipeline
# =========================
model = LinearRegression()

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================
# 3. Train model
# =========================
pipeline.fit(X_train, y_train)


# =========================
# 4. Evaluate model
# =========================
y_pred = pipeline.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")


# =========================
# 5. Save trained model
# =========================
joblib.dump(pipeline, "../models/linear_regression_model.pkl")
print("Model saved successfully")