import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib
from flask import Flask, request, jsonify

# Step 1: Load the Dataset
data_path = "AI_ML_Internship_Problems.csv" 
data = pd.read_csv(data_path)

# Step 2: Data Cleaning
# Check for missing values
# print("Missing Values:")
# print(data.isnull().sum())

# Handle missing values (example: drop rows with missing values)
data_cleaned = data.dropna().copy() # Add .copy() to ensure it's a separate object
# Verify if null values are dropped
# print("Null values after cleaning:")
# print(data_cleaned.isnull().sum())

# Drop the unwanted columns
data_cleaned = data_cleaned.drop(columns=['Shipment ID', 'Origin', 'Destination', 'Shipment Date', 'Planned Delivery Date', 'Actual Delivery Date'])
# print(data_cleaned.head())

# Remove duplicates
data_cleaned = data_cleaned.drop_duplicates()
# Verify the cleaned data
# print(data_cleaned.head())

# Check for duplicate rows
duplicates = data_cleaned.duplicated().sum()
# print(f"Number of duplicate rows: {duplicates}")
data_cleaned = data_cleaned.drop_duplicates()

# Step 3: Exploratory Data Analysis (EDA)
# Summary statistics
print("\nSummary Statistics of Cleaned data-:")
print(data_cleaned.describe())
print("\n")


# Define the bins for the distance ranges
bins = [0, 500, 1000, 1500, 2000]
# Define the labels for each bin
labels = [0, 1, 2, 3]
# Apply encoding using pd.cut()
data_cleaned['Distance (km)'] = pd.cut(data_cleaned['Distance (km)'], bins=bins, labels=labels, right=True)
# print(data_cleaned.head())

# Define mappings for encoding
vehicle_mapping = {'Lorry': 0, 'Truck': 1, 'Trailer': 2, 'Container': 3}
weather_mapping = {'Clear': 0, 'Rain': 1, 'Fog': 2, 'Storm': 3}
traffic_mapping = {'Light': 0, 'Moderate': 1, 'Heavy': 2}
delayed_mapping = {'Yes': 0, 'No': 1}

data_cleaned['Vehicle Type'] = data_cleaned['Vehicle Type'].map(vehicle_mapping)
data_cleaned['Weather Conditions'] = data_cleaned['Weather Conditions'].map(weather_mapping)
data_cleaned['Traffic Conditions'] = data_cleaned['Traffic Conditions'].map(traffic_mapping)
data_cleaned['Delayed'] = data_cleaned['Delayed'].map(delayed_mapping)

# print(data_cleaned.head())


# Save cleaned data for model development
data_cleaned.to_csv("cleaned_data.csv", index=False)
print("Cleaned data saved to 'cleaned_data.csv'.")
print("\n")
# print(data_cleaned.head())

# Split the dataset
X = data_cleaned.drop(['Delayed'], axis=1)
y = data_cleaned['Delayed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Experiment with Decision Tree
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)

# Experiment with Random Forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
y_pred_forest = random_forest_model.predict(X_test)
print(data_cleaned.head())
print("\n")


# Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation for {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\n")

if 'y_pred_logistic' in locals():
    evaluate_model(y_test, y_pred_logistic, "Logistic Regression")
    evaluate_model(y_test, y_pred_tree, "Decision Tree")
    evaluate_model(y_test, y_pred_forest, "Random Forest")

# Save the best model
best_model = random_forest_model  # Assuming Random Forest performed best
joblib.dump(best_model, 'shipment_delay_model.pkl')



# Initialize Flask app
app = Flask(__name__)

# Load the best model
bestfit_model = joblib.load('shipment_delay_model.pkl')

# Predict function for API deployment
@app.route('/Arogo_AI_API', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.get_json()

        # Ensure the data contains the required fields
        required_fields = ['Vehicle Type', 'Distance (km)', 'Weather Conditions', 'Traffic Conditions']
        if not all(field in input_data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Define mappings for encoding
        vehicle_mapping = {'Lorry': 0, 'Truck': 1, 'Trailer': 2, 'Container': 3}
        weather_mapping = {'Clear': 0, 'Rain': 1, 'Fog': 2, 'Storm': 3}
        traffic_mapping = {'Light': 0, 'Moderate': 1, 'Heavy': 2}

        # Apply encoding to input data
        try:
            encoded_data = {
                'Vehicle Type': vehicle_mapping[input_data['Vehicle Type']],
                'Distance (km)': int(input_data['Distance (km)']),  # Assuming distance is already numeric
                'Weather Conditions': weather_mapping[input_data['Weather Conditions']],
                'Traffic Conditions': traffic_mapping[input_data['Traffic Conditions']]
            }
        except KeyError as e:
            return jsonify({'error': f'Invalid value provided for {str(e)}. Please check your input.'}), 400

        # Prepare the input data for prediction
        input_df = pd.DataFrame([encoded_data])

        # Ensure no missing values in the input data
        if input_df.isnull().values.any():
            return jsonify({'error': 'Invalid data encoding. Please check your input values.'}), 400

        # Predict the delay (1 = Yes, 0 = No)
        prediction = bestfit_model.predict(input_df)[0]

        # Return the prediction as a JSON response
        delay = 'Delayed' if prediction == 0 else 'On Time'
        return jsonify({'Prediction': delay})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run()

   
