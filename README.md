# 💙 Cardio Health AI  
### Advanced Cardiovascular Disease Prediction System

---

## 🔬 Overview

Cardio Health AI is an end-to-end Machine Learning system designed to predict the risk of cardiovascular disease using patient health data.

Unlike basic prediction systems, this project integrates **Explainable AI (XAI)** to provide:
- 📊 Risk prediction
- 🧠 Key contributing factors
- 💡 Personalized health recommendations

This makes the system more **interpretable, practical, and user-friendly**.

---

## 🚀 Key Features

- ✅ End-to-End ML Pipeline  
- 🧠 Feature Engineering (BMI, BP difference, risk indicators)  
- 🤖 Random Forest Model  
- 📊 Risk Prediction with Score  
- 🧾 Explainable AI (Reason behind prediction)  
- 💡 Health Recommendations  
- 🎨 Premium Dashboard UI (Streamlit)  
- 🔌 Flask API for integration  

---

## 🧠 Machine Learning Workflow

1. Data Collection  
2. Data Preprocessing (Cleaning + Scaling)  
3. Feature Engineering  
4. Model Training (Random Forest)  
5. Model Evaluation  
6. Model Deployment  

---

## 📂 Project Structure
Cardiovascular-Disease-Prediction/
│── data/
│ └── cardio_train.csv
│
│── models/ # Generated after training (not uploaded)
│
│── src/
│ ├── train.py # Model training script
│ ├── predict.py # Prediction logic
│ └── utils.py # Feature engineering
│
│── app.py # Streamlit UI
│── api.py # Flask API
│── requirements.txt
│── README.md



---

## 📊 Application Preview

### 🏥 Patient Input Dashboard
![Input UI](assets/image1.png)

### 📊 Risk Assessment Output
![Prediction](assets/image2.png)

### 🧠 Risk Factors & Insights
![Insights](assets/image3.png)

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/Sujith-04-K/Cardiovascular-Disease-Prediction.git
cd Cardiovascular-Disease-Prediction



---

### 2️⃣ Install Dependencies
pip install -r requirements.txt



---

### 3️⃣ Train the Model
python src/train.py



---

### 4️⃣ Run the Application
streamlit run app.py



---

## 📥 Input Parameters

- Age  
- Gender  
- Height & Weight  
- Blood Pressure (Systolic & Diastolic)  
- Cholesterol Level  
- Glucose Level  
- Smoking  
- Alcohol Intake  
- Physical Activity  

---

## 🧬 Feature Engineering

To improve model performance, additional features were created:

- BMI (Body Mass Index)  
- Blood Pressure Difference  
- Age Groups  
- High BP Indicator  
- High Cholesterol Indicator  

---

## 📈 Output

The system provides:

- ✅ Risk Level (High / Low)  
- 📊 Risk Score (%)  
- 🧠 Risk Factors (Explainable AI)  
- 💡 Health Recommendations  

---

## 🎯 Key Highlight

> This project goes beyond traditional ML models by integrating **Explainable AI**, allowing users to understand *why* a prediction was made rather than just seeing the result.

---

## ⚠️ Notes

- Model is trained on adult data (18+)  
- Model files (`.pkl`) are not included due to GitHub size limits  
- Run `train.py` to generate the model  

---

## 🛠️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Flask  

---

## 🚀 Future Improvements

- Hyperparameter tuning  
- Deep Learning models  
- Real-time data integration  
- Cloud deployment  

---

## 👨‍💻 Author

**Sujith K**  
🔗 GitHub: https://github.com/Sujith-04-K  

---

## ⭐ Support

If you found this project useful, please ⭐ the repository!
