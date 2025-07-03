import streamlit as st
import joblib
import numpy as np

# Load model and preprocessing tools
model = joblib.load('models/heart_model.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
target_encoder = joblib.load('models/target_encoder.pkl')

# Define input fields
st.title("🏥 Heart Disease Risk Predictor")
st.markdown("This tool helps nurses assess if a patient is **at risk** for heart disease based on key clinical indicators.")

# Input form
with st.form("risk_form"):
    col1, col2 = st.columns(2)

    with col1:
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        PhysicalHealth = st.slider("Physical Health (days affected in past 30)", 0, 30, 5)
        MentalHealth = st.slider("Mental Health (days affected in past 30)", 0, 30, 5)
        SleepTime = st.number_input("Sleep Time (hours per day)", min_value=0.0, max_value=24.0, value=7.0)
        Sex = st.selectbox("Sex", ['Male', 'Female'])
        AgeCategory = st.selectbox("Age Category", [
            '18-24', '25-29', '30-34', '35-39', '40-44',
            '45-49', '50-54', '55-59', '60-64', '65-69',
            '70-74', '75-79', '80 or older'
        ])
        Race = st.selectbox("Race", ['White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
                                     'Hispanic', 'Other'])

    with col2:
        Smoking = st.selectbox("Smoking", ['Yes', 'No'])
        AlcoholDrinking = st.selectbox("Alcohol Drinking", ['Yes', 'No'])
        Stroke = st.selectbox("Stroke History", ['Yes', 'No'])
        DiffWalking = st.selectbox("Difficulty Walking", ['Yes', 'No'])
        Diabetic = st.selectbox("Diabetic", ['Yes', 'No', 'No, borderline diabetes', 'Yes (during pregnancy)'])
        PhysicalActivity = st.selectbox("Physical Activity", ['Yes', 'No'])
        GenHealth = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
        Asthma = st.selectbox("Asthma", ['Yes', 'No'])
        KidneyDisease = st.selectbox("Kidney Disease", ['Yes', 'No'])
        SkinCancer = st.selectbox("Skin Cancer", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict Risk")

# Run prediction
if submitted:
    # Collect inputs in same order as training
    input_dict = {
        'BMI': BMI,
        'PhysicalHealth': PhysicalHealth,
        'MentalHealth': MentalHealth,
        'SleepTime': SleepTime,
        'Smoking': Smoking,
        'AlcoholDrinking': AlcoholDrinking,
        'Stroke': Stroke,
        'DiffWalking': DiffWalking,
        'Sex': Sex,
        'AgeCategory': AgeCategory,
        'Race': Race,
        'Diabetic': Diabetic,
        'PhysicalActivity': PhysicalActivity,
        'GenHealth': GenHealth,
        'Asthma': Asthma,
        'KidneyDisease': KidneyDisease,
        'SkinCancer': SkinCancer
    }

    numerical_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    categorical_cols = [col for col in input_dict if col not in numerical_cols]

    # Encode categorical values
    for col in categorical_cols:
        encoder = label_encoders[col]
        input_dict[col] = encoder.transform([input_dict[col]])[0]

    # Prepare feature array in training column order
    input_array = np.array([[input_dict[col] for col in model.feature_names_in_]])

    # Get numerical column indices (like in train.py)
    numerical_indices = [list(model.feature_names_in_).index(col) for col in numerical_cols]

    # Scale only the numerical features
    input_array[:, numerical_indices] = scaler.transform(input_array[:, numerical_indices])

    # Predict
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1] 

    result = target_encoder.inverse_transform([prediction])[0]

    st.subheader("🩺 Prediction Result")
    if result == 'Yes':
        st.error("⚠️ The patient is **At Risk** of heart disease.")
    else:
        st.success("✅ The patient is **Not at Risk** of heart disease.")

    st.info(f"💡 Confidence Score: **{probability:.2%}** for being at risk.")

