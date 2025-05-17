import streamlit as st
import pandas as pd
import joblib

# App Title & Setup
st.set_page_config(page_title="Glucose Level Predictor", layout="wide")
st.title("🔬 Glucose Level Prediction App")
st.markdown("Predict a patient's **Glucose Level** based on their health and lifestyle inputs.")

# Load models
model_files = {
    "LightGBM": "best_model_lightgbm.pkl",
    "Random Forest": "best_model_random_forest.pkl",
    "XGBoost": "best_model_xgboost.pkl"
}

model_scores = pd.DataFrame({
    'Model': ['LightGBM', 'Random Forest', 'XGBoost'],
    'MAE': [1.48, 4.09, 2.18],  
    'RMSE': [1.95, 5.38, 2.85],
    'R²': [0.99, 0.95, 0.99]
})

with st.expander("📈 View Model Performance Summary"):
    st.dataframe(model_scores)

model_choice = st.selectbox("📊 Select Model", list(model_files.keys()))
model = joblib.load(model_files[model_choice])

# Form for Prediction
st.subheader("🧍 Patient Details")
with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 10, 100, 30)
        carbohydrate_intake = st.selectbox("Carbohydrate Intake", ['Low', 'Medium', 'High'])
        insulin_dosage = st.number_input("Insulin Dosage (units)", 0.0, 100.0, 20.0)

    with col2:
        bmi = st.number_input("BMI", 10.0, 60.0, 24.5)
        exercise_level = st.selectbox("Exercise Level", ['Sedentary', 'Moderate', 'Active'])
        bp = st.slider("Blood Pressure (Systolic)", 70, 200, 120)

    with col3:
        sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
        stress_level = st.selectbox("Stress Level", ['Low', 'Moderate', 'High'])

    submitted = st.form_submit_button("🔮 Predict")

# Recommendation Logic
def get_personalized_recommendations(glucose_level, bmi, sleep_duration, exercise_level):
    recs = []

    # Glucose Level
    if glucose_level < 70:
        recs.append("⚠️ Your glucose level is low. Include fast-acting carbs like juice or glucose tablets and consult your doctor.")
    elif 70 <= glucose_level <= 140:
        recs.append("✅ Your glucose level is normal. Maintain healthy lifestyle habits.")
    elif 140 < glucose_level <= 180:
        recs.append("⚠️ Slightly elevated glucose. Reduce sugar intake, eat more fiber, and stay active.")
    else:
        recs.append("🚨 High glucose level. Seek medical advice and follow your diabetes management plan strictly.")

    # BMI
    if bmi < 18.5:
        recs.append("🍽️ Your BMI suggests you're underweight. Consider nutrient-rich calorie intake with guidance from a nutritionist.")
    elif 18.5 <= bmi <= 24.9:
        recs.append("✅ Your BMI is in the healthy range. Keep up your current routine!")
    elif 25 <= bmi <= 29.9:
        recs.append("⚠️ You are slightly overweight. Moderate daily activity and a lower-calorie diet are recommended.")
    else:
        recs.append("🚨 High BMI. Prioritize weight loss through consistent physical activity and a balanced diet.")

    # Sleep Duration
    if sleep_duration < 6:
        recs.append("🛌 You are sleeping less than recommended. Aim for 7–9 hours to support glucose regulation.")
    elif sleep_duration > 9:
        recs.append("💤 You’re oversleeping. This can sometimes affect metabolism. Try to keep it within 7–9 hours.")
    else:
        recs.append("✅ Great! Your sleep duration is within the ideal range.")

    # Exercise Level
    if exercise_level == 'Sedentary':
        recs.append("🏃 Try to increase physical activity. Even light exercise helps regulate blood sugar.")
    elif exercise_level == 'Moderate':
        recs.append("👍 Your exercise level is fair. Consider incorporating more consistent cardio or resistance training.")
    else:  # Active
        recs.append("✅ Excellent! Your activity level supports healthy metabolism and glucose control.")

    return "### 📌 Recommendations:\n" + "\n".join([f"- {rec}" for rec in recs])

# Predict and show recommendations
if submitted:
    input_data = pd.DataFrame([{
        'Age': age,
        'BMI': bmi,
        'Carbohydrate_Intake': carbohydrate_intake,
        'Exercise_Level': exercise_level,
        'Insulin_Dosage': insulin_dosage,
        'BP': bp,
        'Sleep_Duration': sleep_duration,
        'Stress_Level': stress_level
    }])

    prediction = model.predict(input_data)[0]
    st.success(f"🔮 Predicted Glucose Level: **{prediction:.2f} mg/dL**")

    recommendations = get_personalized_recommendations(prediction, bmi, sleep_duration, exercise_level)
    st.markdown(recommendations)

# Batch prediction
st.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with 8 features (no Glucose column)", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        preds = model.predict(df)
        df['Predicted_Glucose'] = preds
        st.success("✅ Batch prediction completed.")
        st.dataframe(df.head())
        st.download_button("📥 Download Batch Results", df.to_csv(index=False), "batch_predictions.csv")
    except Exception as e:
        st.error(f"⚠️ Error in prediction: {e}")

# Footer
st.markdown("---")
st.markdown("<center>Designed by <b>Aniete</b> as a <i>research project</i>.</center>", unsafe_allow_html=True)
