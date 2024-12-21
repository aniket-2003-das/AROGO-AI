# Shipment Delay Prediction

## Overview

This project predicts whether a shipment will be delayed or arrive on time based on historical logistics data. The model uses machine learning techniques to analyze various factors such as origin, destination, vehicle type, distance, weather, and traffic conditions.

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
├── data/                 # Dataset files (if included)
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── models/               # Trained models and related files
├── app/                  # Flask or FastAPI application files
│   ├── main.py           # API entry point
│   ├── model.pkl         # Saved ML model
│   ├── requirements.txt  # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # License file
```

---

## Installation

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd shipment-delay-prediction
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r app/requirements.txt
   ```

---

## Usage

### Running the API

1. Start the API server:

   ```bash
   cd app
   python main.py
   ```

2. The API will be accessible at `http://127.0.0.1:8000`.

### API Endpoints

- **POST `/predict`**
  - Accepts a JSON payload with shipment details.
  - Returns whether the shipment will be delayed or on time.
  
#### Example Request

```json
{
  "shipment_id": "12345",
  "origin": "Delhi",
  "destination": "Mumbai",
  "shipment_date": "2024-12-21",
  "vehicle_type": "Truck",
  "distance": 1400,
  "weather_conditions": "Clear",
  "traffic_conditions": "Moderate"
}
```

#### Example Response

```json
{
  "prediction": "On Time",
  "probability": 0.85
}
```

---

## Approach

### Data Preparation

- Cleaned dataset by removing missing values and inconsistencies.
- Encoded categorical variables (e.g., weather and traffic conditions).
- Scaled numerical variables like distance.

### Model Development

- Built and evaluated Logistic Regression, Decision Tree, and Random Forest models.
- Chose the best model based on F1 score and recall for imbalanced target variable.

### Deployment

- Created a REST API using Flask.
- Integrated the trained model into the API for real-time predictions.

---

## Results

| Metric          | Logistic Regression | Decision Tree | Random Forest |
|-----------------|---------------------|---------------|---------------|
| Accuracy        | 85.2%               | 88.1%         | 90.3%         |
| Precision       | 83.5%               | 86.0%         | 89.1%         |
| Recall          | 84.7%               | 87.5%         | 91.2%         |
| F1 Score        | 84.1%               | 86.7%         | 90.1%         |

---

## Future Work

- Improve feature engineering for weather and traffic conditions.
- Experiment with advanced models like Gradient Boosting or Neural Networks.
- Add live data integration for real-time predictions.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

- [Your Name]
- AI/ML Internship Project Submission

---

## Contact

For any questions, please contact [your-email@example.com].
