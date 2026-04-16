import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix



data = pd.read_csv("C:/Users/harsh/Downloads/india_air_quality_12000.csv")

print("Data loaded!")
print("Dataset size:", data.shape)
print("\nPreview:\n", data.head())
print("\nColumns:", data.columns.tolist())
print("\nData Types:\n", data.dtypes)
print("\nStatistical Summary:\n", data.describe())
print("\nMissing Values:\n", data.isnull().sum())
print("\nDuplicate Rows:", data.duplicated().sum())
print("\nUnique Cities:", data["City"].nunique())
print("Unique States:", data["State"].nunique())
print("\nAQI Category Count:\n", data["AQI_Bucket"].value_counts())

data["Date"] = pd.to_datetime(data["Date"])
data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month
def assign_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

data["Season"] = data["Month"].apply(assign_season)
print("\nBefore cleaning:", data.shape)
data = data.drop_duplicates().reset_index(drop=True)
print("After removing duplicates:", data.shape)
pollutants = ['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
for feature in pollutants:
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    prev_count = data.shape[0]
    data = data[(data[feature] >= low) & (data[feature] <= high)]
    print(f"{feature}: removed {prev_count - data.shape[0]} values")
data = data.reset_index(drop=True)
print(" dataset size:", data.shape)
data.to_csv("air_quality.csv", index=False)
encoder = LabelEncoder()
data["City_code"] = encoder.fit_transform(data["City"])
data["State_code"] = encoder.fit_transform(data["State"])
data["Zone_code"] = encoder.fit_transform(data["Zone"])
data["Season_code"] = encoder.fit_transform(data["Season"])

data["AQI_class"] = encoder.fit_transform(data["AQI_Bucket"])

print("\nEncoded AQI Labels:\n", data[["AQI_Bucket", "AQI_class"]].value_counts())

input_features = ['PM2_5','PM10','NO2','SO2','CO','O3',
                  'City_code','Zone_code','Month','Season_code']

X = data[input_features]

y_regression = data["AQI"]
y_classification = data["AQI_class"]
X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_regression, test_size=0.2, random_state=42)
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_classification, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nScaling done!")

plt.figure()
plt.hist(data["AQI"], bins=40)
plt.title("AQI Distribution")
plt.xlabel("AQI")
plt.ylabel("Frequency")
plt.show()
plt.figure()
data["AQI_Bucket"].value_counts().plot(kind='bar')
plt.title("AQI Categories")
plt.show()
plt.figure()
sns.heatmap(data[pollutants].corr(), annot=True)
plt.title("Correlation Matrix")
plt.show()
plt.figure()
data.groupby("City")["AQI"].mean().sort_values().plot(kind='barh')
plt.title("City-wise AQI")
plt.show()
plt.figure()
sns.boxplot(x="Season", y="AQI", data=data)
plt.title("AQI by Season")
plt.show()
plt.figure()
plt.scatter(data["PM2_5"], data["AQI"])
plt.title("PM2.5 vs AQI")
plt.show()
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train_r)
predicted_aqi = lin_model.predict(X_test_scaled)
mse_value = mean_squared_error(y_test_r, predicted_aqi)
r2_value = r2_score(y_test_r, predicted_aqi)
print("\n--- Linear Regression ---")
print("MSE:", round(mse_value, 2))
print("RMSE:", round(np.sqrt(mse_value), 2))
print("R2 Score:", round(r2_value, 4))
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train_c)
predicted_class = rf_model.predict(X_test)
accuracy = accuracy_score(y_test_c, predicted_class)
print("\n--- Random Forest ---")
print("Accuracy:", round(accuracy, 4))
print(classification_report(y_test_c, predicted_class))
plt.figure()
plt.scatter(y_test_r, predicted_aqi)
plt.title("Actual vs Predicted AQI")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
cmatrix = confusion_matrix(y_test_c, predicted_class)
plt.figure()
sns.heatmap(cmatrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
importance = pd.Series(rf_model.feature_importances_, index=input_features)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()