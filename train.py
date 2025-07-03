import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

df = pd.read_csv("dataset/heart_2020_uncleaned.csv")

numerical_cols = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
for col in numerical_cols:
    if df[col].isnull().any():
        median_value = df[col].median()
        df.fillna({col: median_value}, inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        mode_value = df[col].mode()[0]
        df.fillna({col: mode_value}, inplace=True)

label_encoders = {}
categorical_features = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 
                       'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',
                       'Asthma', 'KidneyDisease', 'SkinCancer']

for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

target_encoder = LabelEncoder()
df['HeartDisease'] = target_encoder.fit_transform(df['HeartDisease'])

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numerical_indices = [X.columns.get_loc(col) for col in numerical_cols]
X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

try:
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/heart_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    print("Model successfuly dumped!")
except Exception as e:
    print("Error occured: ", e)
    
