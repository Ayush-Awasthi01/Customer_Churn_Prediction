import os
import pandas as pd
import joblib
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -------------------------
# CONFIGURATION
# -------------------------
DATA_PATH = r"C:\Users\Ayush Awasthi\Desktop\Churn_predication\tel_churn.csv"
MODEL_PATH = "model.sav"

app = Flask(__name__)

# -------------------------
# FUNCTION TO TRAIN MODEL
# -------------------------
def train_and_save_model():
    print("📌 Training model...")

    df = pd.read_csv(DATA_PATH)

    # Split features and target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )

    grid = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print("✅ Best Hyperparameters:", grid.best_params_)
    print("✅ Test Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model and feature names
    joblib.dump({
        'model': best_model,
        'features': X_train.columns.tolist()
    }, MODEL_PATH)

    print(f"✅ Model trained and saved as {MODEL_PATH}")
    return best_model, X_train.columns.tolist()

# -------------------------
# LOAD MODEL (or train if missing)
# -------------------------
if os.path.exists(MODEL_PATH):
    print("📌 Loading existing model...")
    model_and_features = joblib.load(MODEL_PATH)
    model = model_and_features['model']
    expected_features = model_and_features['features']
else:
    model, expected_features = train_and_save_model()

# -------------------------
# LOAD BASE DATA
# -------------------------
try:
    df_base = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    exit(f"Error: Dataset not found at {DATA_PATH}")

# -------------------------
# FLASK ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def load_page():
    return render_template("home.html")

@app.route("/", methods=['POST'])
def predict():
    form = request.form
    input_data = {
        'SeniorCitizen': form.get('query1', ''),
        'MonthlyCharges': form.get('query2', ''),
        'TotalCharges': form.get('query3', ''),
        'gender': form.get('query4', ''),
        'Partner': form.get('query5', ''),
        'Dependents': form.get('query6', ''),
        'PhoneService': form.get('query7', ''),
        'MultipleLines': form.get('query8', ''),
        'InternetService': form.get('query9', ''),
        'OnlineSecurity': form.get('query10', ''),
        'OnlineBackup': form.get('query11', ''),
        'DeviceProtection': form.get('query12', ''),
        'TechSupport': form.get('query13', ''),
        'StreamingTV': form.get('query14', ''),
        'StreamingMovies': form.get('query15', ''),
        'Contract': form.get('query16', ''),
        'PaperlessBilling': form.get('query17', ''),
        'PaymentMethod': form.get('query18', ''),
        'tenure': form.get('query19', ''),
    }

    new_df = pd.DataFrame([input_data])

    # Convert numeric fields
    for col in ["TotalCharges", "MonthlyCharges", "tenure", "SeniorCitizen"]:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0.0)

    # Combine for encoding
    combined_df = pd.concat([df_base.copy(), new_df], ignore_index=True, sort=False)

    bins = [0, 12, 24, 36, 48, 60, 72, float('inf')]
    labels = ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72", "73+"]
    combined_df['tenure_group'] = pd.cut(combined_df['tenure'], bins=bins, right=False, labels=labels, include_lowest=True)
    combined_df.drop(columns=['tenure'], inplace=True)

    cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']

    dummies = pd.get_dummies(combined_df[cat_cols])

    for num_col in ['MonthlyCharges', 'TotalCharges']:
        dummies[num_col] = combined_df[num_col]

    X = dummies.tail(1)
    X = X.reindex(columns=expected_features, fill_value=0)

    pred_class = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0][pred_class]

    output_text = "This customer is likely to churn!" if pred_class == 1 else "This customer is likely to continue!"
    confidence_text = f"Confidence: {pred_proba*100:.2f}%"

    return render_template(
        "home.html",
        output1=output_text,
        output2=confidence_text,
        **{f'query{i}': form.get(f'query{i}','') for i in range(1, 20)}
    )

# -------------------------
# RUN FLASK
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
