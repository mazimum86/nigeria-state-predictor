# 🌍 Nigerian State Predictor Using Latitude & Longitude

This Streamlit web app uses a trained Artificial Neural Network (ANN) to predict the Nigerian state for any given **latitude and longitude**. It also returns the **top 3 most probable states** and their associated prediction confidence.

> 🎯 Built for data-driven geospatial classification using major telecoms and collocation partner datasets across Nigeria.

---

## 📌 Demo

🔗 Live App: [Streamlit Cloud Deployment](https://mazimum86-nigeria-state-predictor.streamlit.app)

---

## ✨ Features

- 📍 Predicts one of **37 Nigerian regions** (36 states + FCT renamed to Abuja)
- 🔢 Uses a trained ANN (`main_model.keras`) and scaled features via `scaler.pkl`
- 📊 Shows top 3 predictions with probabilities
- 🧠 Trained on telecom infrastructure geocoordinates
- 🌐 Powered by TensorFlow, Streamlit, and Scikit-learn

---

## 🧠 How It Works

### Input:
- Latitude (e.g., `9.0578`)
- Longitude (e.g., `7.4951`)

### Output:
- Most likely state: `Abuja`
- Probability: `92.4%`
- Other candidates: `Nasarawa (6.3%)`, `Kogi (1.3%)`

---

## 🛠️ Setup Locally

```bash
# Clone the repo
git clone https://github.com/mazimum86/nigeria-state-predictor.git
cd nigeria-state-predictor

# Create and activate a virtual environment
conda create -n streamlit-env python=3.10
conda activate streamlit-env

# Install required libraries
pip install -r requirements.txt

# Run the app
streamlit run main.py
```

## 📁 Project Structure
```
📦 nigeria-state-predictor/
┣ 📜main.py                        
┣ 📜requirements.txt
┣ 📜runtime.txt
┣ 📜scaler.pkl
┣ 📜main_model.keras
┗ 🖼️screenshot.png      
├── .streamlit/
│   └── config.toml       
└── README.md              


```
## 📸 Screenshots
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/48bd23c4-39f2-4029-b295-cc4a2e5629cb" />


## 📤 Deployment
App is deployed on Streamlit Cloud.
Free to use, open source, and no signup required for visitors!

## 🙋‍♂️ Creator Info
🔧 Created by: Chukwuka Chijioke Jerry
📧 Email: chukwuka.jerry@gmail.com
📱 WhatsApp: +2348038782912
🔗 LinkedIn: linkedin.com/in/chukwukacj
🐦 X (Twitter): @Mazimum_

🏁 Future Work
📦 Add support for batch predictions via file upload

🌍 Extend model to include neighboring countries

🧠 Explore CNNs and RNNs for spatial data encoding

## 💡 License
This project is open-source under the MIT License.



