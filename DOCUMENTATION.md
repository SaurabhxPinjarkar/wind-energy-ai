# Wind Energy Potential Prediction System: Complete Data Flow & Architecture

This document outlines the complete lifecycle of data within the application, starting from historical data ingestion to the final real-time prediction output on the user interfaces.

---

## 1. External APIs & Data Sources

**Primary Source:** [NASA POWER API](https://power.larc.nasa.gov/) (Prediction Of Worldwide Energy Resources)  
The system relies entirely on NASA’s satellite telemetry and meteorology API for both historical training data and real-time inference data.

*   **Endpoint Used:** `https://power.larc.nasa.gov/api/temporal/daily/point`
*   **Community:** `RE` (Renewable Energy)
*   **Format:** `JSON`

### Fetched Meteorological Parameters:
1.  **WS10M:** Wind Speed at 10 Meters (m/s)
2.  **T2M:** Temperature at 2 Meters (°C)
3.  **PS:** Surface Pressure (kPa / mb)
4.  **RH2M:** Relative Humidity at 2 Meters (%)

---

## 2. Phase A: Data Ingestion & Generation (`DATASET.py` & `fix_coordinates.py`)

To train the Machine Learning model, the system requires a deep historical baseline of weather behavior across Maharashtra.

*   **Location Initialization:** 100 coordinates representing major geographical profiles of Maharashtra (Ratnagiri, Pune, Nagpur, Western Ghats, Deccan Plateau, etc.) with slight random variations (± 0.15 degrees) to simulate real-world scatter mapping.
*   **Timeframe:** `start=20200101` to `end=20231231` (4 full years of daily data).
*   **Process Output:** Fetches all parameters for 1461 days across 100 locations and outputs locally to a static CSV file: `final_wind_dataset.csv` (~131,490 rows).

---

## 3. Phase B: Feature Engineering & Constants (`1_data_preprocessing.py`)

Before training, raw meteorological data is transformed into a Machine Learning-ready state.

### Constants & Mathematical Assertions
*   **Target Variable Formula ($y$):** `Energy = (WS10M) ** 3` 
    *   *Why?* According to wind power physics (Betz's Law and Kinetic Energy equations), the power available in the wind is directly proportional to the cube of the wind velocity.
*   **Feature Columns ($X$):** `['WS10M', 'T2M', 'PS', 'RH2M', 'Latitude', 'Longitude']`.
*   **Standardization Target:** Applied `StandardScaler` to normalize the scales of pressure (values ~100) vs temperature (values ~30) so the AI models process them evenly without weight bias. Outputs `scaler.pkl`.

---

## 4. Phase C: Machine Learning Pipeline (`2_model_training.py`)

The system separates the ~131k rows into an 80% Training Split and a 20% Evaluation Test split.

1.  **Tested Models:** `Linear Regression`, `Random Forest Regressor`, and `XGBoost`.
2.  **Selection Logic:** Evaluated based on Root Mean Square Error (RMSE) and R² Variance.
3.  **Winning Model:** **Random Forest**. It achieved near-perfect variance capturing (`R² = 0.9995`).
4.  **Artifact Generation:** The winning model state is serialized into a binary `model_rf.pkl` file for instant inference.

---

## 5. Phase D: Business Logic Configuration (`3_model_deployment.py`)

This phase dictates the "Suitability" definitions that power the front-end dashboard badges by establishing rigid constants.

*   **Suitability Threshold (The 50th Percentile):**
    *   The system analyzes the generated `Energy` metric from ALL 131,000 historical rows.
    *   It calculates the absolute median (50th percentile) of wind energy generation potential in this region. 
    *   **Condition:** If a real-time prediction calculates an energy yield `> Median`, it is deemed **"Good"**. If `< Median`, it is deemed **"Not Suitable"**. (This threshold sits at approximately `31.86 kWh/unit`).

---

## 6. Phase E: Live Output & Flask Web Server (`4_website.py` & `app.js`)

This is the real-time application flow when the user interacts with the UI Map.

### Standard Request Flow Steps:
1.  **User Interaction:** User clicks precisely on the Leaflet.js interactive map (sending exact `Latitude` & `Longitude` to Flask).
2.  **Internal API Call:** Frontend `app.js` triggers a `POST` request to local endpoint `/api/predict`.
3.  **Live Weather Ingestion:** The Flask Server immediately pings the NASA POWER API for the latest accessible climatology data for those exact coordinates over the previous 7 days (averaging it out to simulate current stable atmospheric conditions).
4.  **Data Transformation:** The raw NASA live telemetry is parsed and passed into the loaded `scaler.pkl` to normalize it identical to Phase B.
5.  **Model Inference:** The normalized NumPy array is passed to `model_rf.pkl` which calculates the final projected `Energy Output`.
6.  **Suitability Wrapping:** The Flask server checks if the returned prediction > `31.86 Threshold` (Phase D) and assigns standard Feasibility percentages.
7.  **Final JSON Payload:** 
    ```json
    {
      "status": "success",
      "predicted_energy": 105.4,
      "suitability_score": 100.0,
      "suitability": "Good",
      "wind_speed": 4.5,
      ...
    }
    ```
8.  **Frontend Render:** JavaScript breaks down the payload and animates it onto the **Strategic Comparison Board**, applying either a green (Good) or red (Threat/Not Suitable) ring.