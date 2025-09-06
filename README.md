<img width="1963" height="605" alt="download" src="https://github.com/user-attachments/assets/f9406ed5-75fd-43ff-9d95-6d16ab903edb" />
# Cardiovascular-Disease-Prediction
Machine Learning project to predict cardiovascular disease using patient health data.

# 🫀 Cardiovascular Disease Prediction (Machine Learning Project)

## 📌 Overview
Cardiovascular disease is one of the leading causes of death worldwide. Early detection can help save lives.  
This project applies **Machine Learning models** on patient health records to predict whether a person is at risk of heart disease.

The project covers **data preprocessing, exploratory data analysis (EDA), model training, evaluation, and deployment (saved model)**.  
The best model achieved **~77% accuracy (Random Forest Classifier)**.

---

## 📊 Dataset
- Source: [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)  
- Size: ~70,000 patient records  
- Features:
  - Age (in days → converted to years)
  - Gender
  - Height, Weight
  - Blood Pressure (`ap_hi`, `ap_lo`)
  - Cholesterol, Glucose
  - Lifestyle (smoking, alcohol, physical activity)
- Target: `cardio`  
  - `0` → No cardiovascular disease  
  - `1` → Cardiovascular disease present  

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy → data processing  
- Matplotlib, Seaborn → visualization  
- Scikit-learn → ML models & evaluation  
- Joblib → model saving  

---

## 📊 Methodology
1. **Data Preprocessing**
   - Converted `age` from days → years
   - Dropped irrelevant column `id`
   - Standardized features (for scaling)

2. **Exploratory Data Analysis (EDA)**
   - Target distribution (`0` vs `1`)
   - Age distribution
   - Correlation matrix between features

3. **Model Training**
   - Logistic Regression  
   - Support Vector Machine (SVM)  
   - K-Nearest Neighbors (KNN)  
   - Decision Tree  
   - Random Forest  

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score
   - Best model: **Random Forest (~77% accuracy)**  

---

## 📈 Results
| Model                | Accuracy |
|-----------------------|----------|
| Logistic Regression   | ~73%     |
| SVM                   | ~74%     |
| KNN                   | ~71%     |
| Decision Tree         | ~70%     |
| **Random Forest**     | **~77%** |

✅ **Random Forest performed the best** — stable, robust, and interpretable.

---

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/cardio-disease-prediction-ml.git
cd cardio-disease-prediction-ml



2.Install Dependencies
pip install -r requirements.txt


3. Run the Notebook
jupyter notebook Cardio_Prediction.ipynb


4. Load the Trained Model
python
import joblib
import pandas as pd

# Load model
model = joblib.load("heart_disease_best_model.pkl")

# Example new patient data
new_patient = pd.DataFrame([{
    "age": 55, "gender": 1, "height": 170, "weight": 80,
    "ap_hi": 145, "ap_lo": 95, "cholesterol": 2, "gluc": 1,
    "smoke": 0, "alco": 0, "active": 1
}])

# Predict
print(model.predict(new_patient))  # 0 = No disease, 1 = Disease




## Project Structure
cardio-disease-prediction-ml/
│── Cardio_Prediction.ipynb      # Jupyter Notebook (all code & analysis)
│── heart_disease_best_model.pkl # Saved ML model
│── requirements.txt             # Dependencies
│── README.md                    # Project Documentation
│── images/                      # Plots & visualizations

