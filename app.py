import streamlit as st
import pandas as pd
import pickle
import numpy as np

# मॉडल लोड करने का फंक्शन (मान लेते हैं कि हमने मॉडल को 'model.pkl' नाम से सेव किया है)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

st.title("⚙️ Predictive Maintenance System")
st.write("मशीन के सेंसर डेटा दर्ज करें ताकि यह पता लगाया जा सके कि मशीन सुरक्षित है या फेल होने का जोखिम है।")

# यूजर से इनपुट लेने के लिए साइडबार या फॉर्म
st.header("🔧 सेंसर पैरामीटर्स इनपुट करें")

air_temp = st.number_input("Air Temperature [K]", value=298.0)
process_temp = st.number_input("Process Temperature [K]", value=308.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=0)

# जब यूजर प्रेडिक्शन बटन दबाए
if st.button("Predict Machine Health"):
    # इनपुट डेटा को मॉडल के फॉर्मेट में तैयार करना
    input_data = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    
    # प्रेडिक्शन करना
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.subheader("📊 परिणाम (Results):")
    if prediction[0] == 1:
        st.error(f"⚠️ चेतावनी: मशीन के फेल होने का खतरा है! (Risk Probability: {prediction_proba[0][1]*:.2f}%)")
    else:
        st.success(f"✅ मशीन बिल्कुल सुरक्षित है और ठीक से काम कर रही है। (Safe Probability: {prediction_proba[0][0]*:.2f}%)")
