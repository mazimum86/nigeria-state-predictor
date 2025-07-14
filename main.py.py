import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Cache model loading
@st.cache_resource
def load_model():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model("main_model.h5")

    return scaler, model

scaler, model = load_model()

# States list
states = sorted([
    'Abia', 'Adamawa', 'Akwa Ibom', 'Anambra', 'Bauchi', 'Bayelsa', 
    'Benue', 'Borno', 'Cross River', 'Delta', 'Ebonyi', 'Edo', 
    'Ekiti', 'Enugu', 'Gombe', 'Imo', 'Jigawa', 'Kaduna', 'Kano', 
    'Katsina', 'Kebbi', 'Kogi', 'Kwara', 'Lagos', 'Nassarawa', 
    'Niger', 'Ogun', 'Ondo', 'Osun', 'Oyo', 'Plateau', 'Rivers', 
    'Sokoto', 'Taraba', 'Yobe', 'Zamfara', 'Abuja'
])

# Page config
st.set_page_config(page_title="Geo-State Predictor", layout="centered")
st.title("🗺️ Nigerian State Predictor")
st.markdown("Enter coordinates to predict the Nigerian state")

# Inputs
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", min_value=4.0, max_value=14.0, 
                         value=9.0820, format="%.6f")
with col2:
    lon = st.number_input("Longitude", min_value=2.0, max_value=15.0, 
                         value=7.4913, format="%.6f")

# Prediction
if st.button("Predict State", type="primary"):
    # Validate input
    if not (4.0 <= lat <= 14.0) or not (2.0 <= lon <= 15.0):
        st.warning("Coordinates outside Nigeria's bounds")
        st.stop()
    
    try:
        # Predict
        scaled = scaler.transform([[lat, lon]])
        probs = tf.nn.softmax(model.predict(scaled)[0]).numpy()
        top_3 = sorted(zip(states, probs*100), key=lambda x: -x[1])[:3]
        
        # Display
        st.success(f"**Predicted State:** {top_3[0][0]} ({top_3[0][1]:.1f}% confidence)")
        
        cols = st.columns(3)
        for idx, (state, prob) in enumerate(top_3):
            with cols[idx]:
                st.metric(
                    label=f"{['🥇','🥈','🥉'][idx]} Prediction",
                    value=state,
                    delta=f"{prob:.1f}%"
                )
                st.progress(int(prob))
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
Built with TensorFlow | Model: main_model.keras  
Developed by [Chijioke Jerry](https://www.linkedin.com/in/chukwukacj)  
GitHub: [@mazimum86](https://github.com/mazimum86)  
""")
