import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
diabetes_dataset = pd.read_csv("diabetes.csv")

# Display the first few rows of the dataset
print(diabetes_dataset.head())

# Display the number of rows and columns in the dataset
print("Dataset shape:", diabetes_dataset.shape)

# Display statistical measures of the dataset
print(diabetes_dataset.describe())

# Display the number of diabetic and non-diabetic persons
print(diabetes_dataset['Outcome'].value_counts())

# Display mean values for all the columns grouped by 'Outcome'
print(diabetes_dataset.groupby('Outcome').mean())

# Separate the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print("Standard")
print(standardized_data)

# Update features with standardized data
X = standardized_data
print(X)
print(Y)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print("Total data:", X.shape)
print("Train data:", X_train.shape)
print("Test data:", X_test.shape)

# Initialize the SVM classifier with a linear kernel
classifier = svm.SVC(kernel='poly', degree = 3)

# Train the SVM classifier
classifier.fit(X_train, Y_train)

# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Accuracy on training data:", training_data_accuracy)

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on test data:", test_data_accuracy)

"""
# Test with new input data
input_data = (3, 158, 76, 36, 245, 31.6, 0.851, 28)

# Convert input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardize the input data as the model was trained with standardized data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

# Make a prediction
prediction = classifier.predict(std_data)
print(prediction)

print("The given data is: ", input_data)

# Output the result
if prediction[0] == 0:
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")
"""