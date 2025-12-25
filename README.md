# Heart Disease Prediction Pipeline 🫀

This project implements an end-to-end Machine Learning pipeline to predict the presence of heart disease using the UCI Heart Disease dataset.

## 🛠️ Machine Learning Principles Applied
- **Binary Classification:** Simplified multi-class diagnosis into a binary "Healthy vs. Sick" model.
- **Data Preprocessing:** - Handled missing values (`?`) using `SimpleImputer`.
    - Automated feature scaling for numerical data using `StandardScaler`.
    - Encoded categorical variables using `OneHotEncoder`.
- **Pipeline Architecture:** Used Scikit-Learn `Pipeline` and `ColumnTransformer` to ensure clean, reproducible data transformations.
- **Model:** Utilized a `RandomForestClassifier` (100 estimators) to capture non-linear relationships in medical data.

## 📋 Data Dictionary & Medical Context

The dataset contains 14 key variables used by clinicians to assess cardiovascular health.

**Demographic & Clinical Vitals**

age: Patient age in years.

sex: (1 = Male; 0 = Female).

trestbps: Resting blood pressure (mm Hg). Values > 120 can indicate hypertension.

chol: Serum cholesterol (mg/dl). High levels (> 200) contribute to artery blockages.

fbs: Fasting blood sugar. (1 if > 120 mg/dl; 0 otherwise). High sugar is a risk factor for diabetes-related heart damage.

**Exercise Stress Test Results**
thalach: Maximum heart rate achieved during exercise. Lower values relative to age can indicate heart weakness.

exang: Exercise-induced angina (1 = Yes; 0 = No). Chest pain brought on by physical stress.

oldpeak: ST depression induced by exercise relative to rest. Measures "stress" on the heart's electrical system; values > 1.0 are clinically significant.

slope: The slope of the peak exercise ST segment (1: Upsloping; 2: Flat; 3: Downsloping).

**Diagnostic Indicators**
cp: Chest pain type (1: Typical Angina; 2: Atypical Angina; 3: Non-anginal pain; 4: Asymptomatic).

restecg: Resting electrocardiographic results (0: Normal; 1: ST-T wave abnormality; 2: Left ventricular hypertrophy).

ca: Number of major vessels (0–3) colored by fluoroscopy. Indicates the presence of visible blockages.

thal: A blood disorder called thalassemia (3: Normal; 6: Fixed defect; 7: Reversible defect).


## 📊 Results
- **Mean Absolute Error (MAE):** 0.17391304347826086
- **Key Predictors:** The model identified Chest Pain type (`cp`), Maximum Heart Rate (`thalach`), and ST Depression (`oldpeak`) as the most significant features.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python "FIRST PROJECT.py"` to train the model and see results.