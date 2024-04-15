# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

# Loading the dataset
df = pd.read_csv('framingham.csv')
df.columns = ['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose','TenYearCHD']

# Data cleaning and manipulation
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df = pd.DataFrame(imputer.fit_transform(df), columns = df.columns)

# Splitting data into training and testing sets
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Evaluating the model
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)
print("Accuracy score: ", accuracy_score(y_test, y_pred))

# Creating a new data point to test the model
# Inputs can also be used as in Arrays
new_data = np.array([
    [1, 50, 2, 1, 15, 0, 0, 0, 0, 200, 120, 80, 25, 70, 100],
    [0, 65, 3, 0, 0, 1, 0, 1, 0, 250, 160, 90, 30, 80, 200],
    [1, 40, 1, 1, 10, 0, 0, 0, 0, 180, 110, 70, 20, 60, 50],
    [0, 55, 2, 0, 0, 0, 0, 1, 1, 220, 140, 85, 27, 90, 150]
])

# Feature scaling for the new data point
new_data_scaled = sc.transform(new_data)
print(new_data)
print(new_data_scaled)
# Making a prediction with the new data point
predictions = classifier.predict(new_data_scaled)

for pred in predictions:
    if pred == 1:
        print("The model predicts that the person has a 10-year risk of developing coronary heart disease.")
    else:
        print("The model predicts that the person does not have a 10-year risk of developing coronary heart disease.")

# Printing all predictions
print(predictions)
