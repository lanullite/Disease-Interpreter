import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import spacy
import joblib
from flask import Flask, request, jsonify
import shap

# Load NLP model
nlp = spacy.load("en_core_web_md")

# Load dataset (replace with real medical dataset)
data = {
    'symptoms': [
        "fever,cough,headache",
        "fatigue,nausea,vomiting",
        "rash,itching,redness",
        "chest_pain,shortness_of_breath",
        "fever,body_ache,sore_throat"
    ],
    'disease': [
        "flu",
        "food_poisoning",
        "allergy",
        "heart_disease",
        "strep_throat"
    ]
}
df = pd.DataFrame(data)

# Preprocess symptoms into vectors
def preprocess_symptoms(symptoms):
    symptom_list = [s.strip() for s in symptoms.split(",")]
    vector = np.mean([nlp(symptom).vector for symptom in symptom_list], axis=0)
    return vector

# Convert symptoms to feature vectors
X = np.array([preprocess_symptoms(s) for s in df['symptoms']])
y = df['disease']

# Encode diseases
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train Random Forest & XGBoost models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Evaluate models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_pred):.2f}")

# Save models
joblib.dump(rf_model, 'rf_disease_model.pkl')
joblib.dump(xgb_model, 'xgb_disease_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# SHAP explainability
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=[f"symptom_{i}" for i in range(X.shape[1])])

# Flask API for predictions
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_disease():
    data = request.json
    symptoms = data['symptoms']
    symptom_vector = preprocess_symptoms(symptoms)
    prediction = rf_model.predict([symptom_vector])[0]
    disease = label_encoder.inverse_transform([prediction])[0]
    
    # Explain prediction using SHAP
    shap_values = explainer.shap_values([symptom_vector])
    explanation = {
        "disease": disease,
        "symptoms": symptoms.split(","),
        "shap_values": shap_values[0].tolist()
    }
    
    return jsonify(explanation)

if __name__ == '__main__':
    app.run(debug=True)