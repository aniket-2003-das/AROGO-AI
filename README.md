# Shipment Delay Prediction

## Overview

This project predicts whether a shipment will be delayed or arrive on time based on historical logistics data. The model uses machine learning techniques to analyze various factors such as origin, destination, vehicle type, distance, weather, and traffic conditions.
Deployed the Flask API Endpoint accessible <https://arogo-ai-api.onrender.com/Arogo_AI_API>

---

## Objectives

1. **Data Preparation & Exploration:**
   - Cleaning and handling missing values.
   - Performing exploratory data analysis (EDA) to identify useful features.
2. **Model Development:**
   - Building and evaluating classification models.
   - Experimenting with at least two machine learning algorithms (Logistic Regression, Decision Tree, Random Forest, etc.).
   - Using metrics like accuracy, precision, recall, and F1 score for evaluation.
3. **Deployment:**
   - Creating a REST API using Flask or FastAPI.
   - Accepting shipment details via API and returning predictions (Delayed/On Time).
4. **Documentation:**
   - Detailing the approach to data preparation and model selection.
   - Explaining API functionality and usage.

---

## Features

- **Inputs:**
  - Shipment ID
  - Origin (Indian cities)
  - Destination (Indian cities)
  - Shipment Date
  - Vehicle Type (Truck, Lorry, Container, Trailer)
  - Distance (km)
  - Weather Conditions (Clear, Rain, Fog)
  - Traffic Conditions (Light, Moderate, Heavy)
- **Output:**
  - Delay (Yes/No)

---

## File Structure

```plaintext
shipment-delay-prediction/
├── AI_ML_Internship_Problems.csv   # Dataset files
├── Shipment Delay Prediction.ipynb # Jupyter notebooks for EDA and modeling
├── shipment_delay_model.pkl        # Trained models
├── app.py                          # Flask API 
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── LICENSE                         # License file
```

---

## Installation

1. Clone this repository:

   ```bash
   git clone <https://github.com/aniket-2003-das/AROGO-AI.git>
   cd shipment-delay-prediction
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/Scripts/activate
   ```

3. Install required packages:

   ```bash
   pip install -r app/requirements.txt
   ```

---

## Implementation

### Running the API Locally

1. Start the API server:

   ```bash
   cd app
   python main.py
   ```

2. The API will be accessible at <http://127.0.0.1:8000/Arogo_AI_API>

### API Endpoints

- **POST `/predict`**
  - Accepts a JSON payload with shipment details.
  - Returns whether the shipment will be delayed or on time.
  
#### Example Request

```json
{
  "vehicle_type": "Truck",
  "distance": 1400,
  "weather_conditions": "Clear",
  "traffic_conditions": "Moderate"
}
```

#### Example Response

```json
{
  "Prediction": "On Time",
}
```

---

## Solution Approach

### Data Preparation

- Cleaned dataset by removing missing values and inconsistencies.
- Encoded categorical variables (e.g., weather and traffic conditions).
- custom categorized numerical variables like distance.
- Droped irrelevant variables like Shipment IDs and dates.

### Model Development

- Built and evaluated Logistic Regression, Decision Tree, and Random Forest models.
- The best model based on F1 score and recall for imbalanced target variable is choosed.

### Deployment

- Created a REST API using Flask.
- Integrated the trained model into the API for real-time predictions.
- Deployed the Flask API Endpoint on <https://arogo-ai-api.onrender.com/Arogo_AI_API>

---

## Results

| Metric          | Model.pkl           |
|-----------------|---------------------|
| Accuracy        | 90.6%               |
| Precision       | 71.4%               |
| Recall          |  100%               |
| F1 Score        | 83.3%               |

---

## Future Work

- Improve feature engineering based on Origin and Destination.
- Experiment with advanced models like Gradient Boosting or Neural Networks.
- Add live data integration for real-time predictions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

- [Aniket Das]
- AI/ML Internship Project Submission for Arogo AI

---

## Contact

For any questions, please contact [aniketdas8822@gmail.com].
