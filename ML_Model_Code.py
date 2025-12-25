import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error

heart_data=pd.read_csv( r"C:\Users\bousm\Downloads\heart.disease.data\heart_disease_uci.csv")
#view our data
print(heart_data)
print(heart_data.describe())
print(heart_data.columns)


# Select target
y = heart_data.num
# Drop unhelpful and target columns
X = heart_data.drop(['num', 'id', 'dataset'], axis=1)
# Check the count of missing values per column
missing_values = heart_data.isnull().sum()

# Calculate the percentage (useful for large datasets)
missing_percent = (heart_data.isnull().sum() / len(heart_data)) * 100

# Combine into a nice table
report = pd.DataFrame({'Total Missing': missing_values, 'Percentage': missing_percent})


print(report[report['Total Missing'] > 0])
# Force these columns to be numeric, turning '?' into NaN
cols_to_fix = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in cols_to_fix:
    heart_data[col] = pd.to_numeric(heart_data[col], errors='coerce')

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

# Get list of categorical variables
s = (X_train.dtypes == 'object')
cat_cols = list(s[s].index)

# Get list of numrical variables
# Manually define - this is more reliable for messy medical data
num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

# print 
print("Categorical variables:")
print(cat_cols)
print("numrical variables:")
print(num_cols)

# 2. Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# 3. Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 4. Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 5. Define the model
model = RandomForestClassifier(n_estimators=100, random_state=0)

# 6. Create and fit the full pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Convert target to binary (0 = healthy, 1 = sick)
y_train_binary = (y_train > 0).astype(int)
y_valid_binary = (y_valid > 0).astype(int)


my_pipeline.fit(X_train, y_train_binary)

# 1. Get predictions
preds = my_pipeline.predict(X_valid)

# 3. Calculate MAE directly

my_mae = mean_absolute_error(y_valid_binary, preds)

print("Mean Absolute Error:", my_mae)

import joblib

# Save the entire pipeline (cleaning steps + model) to a file
joblib.dump(my_pipeline, 'heart_disease_model.pkl')

print("Model saved successfully as 'heart_disease_model.pkl'")