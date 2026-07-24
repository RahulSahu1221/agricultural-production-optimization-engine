# OptiCrop

## Smart Agricultural Production Optimization Engine

OptiCrop is a precise, Machine Learning-based crop recommendation system designed to replace agricultural guesswork with data-driven insights. By analyzing specific soil nutrients and environmental conditions, the engine predicts the most highly suitable crop for cultivation, promoting optimal yields and sustainable farming practices.

Built with a focus on reliability and meticulous code structure, the system seamlessly integrates a robust Python/Scikit-Learn backend with a modern, responsive frontend.

https://agricultural-production-optimization.onrender.com

---

## Key Features

*   **Robust Machine Learning Pipeline:** Automatically preprocesses data (handling missing values, scaling features, and encoding labels) and evaluates multiple algorithms to select the most accurate model.
*   **Expanded Agronomic Dataset:** Features a dynamic data injection script (`expand_dataset.py`) that expands the training data to cover **64 distinct global crops** across 6,400 statistically balanced samples.
*   **Strict Input Validation:** The backend rigorously validates user inputs to ensure all environmental parameters fall within realistic, scientifically accurate ranges before processing predictions.
*   **Premium UI/UX:** A fully responsive, minimal interface built with Bootstrap 5. It features glassmorphism styling, ambient background animations, and smooth light/dark mode toggling.

---

## Technologies Used

### Backend & Machine Learning
*   **Python:** Core programming language.
*   **Flask:** Lightweight WSGI web application framework for serving the API.
*   **Scikit-Learn:** Utilized for the `RandomForestClassifier`, `StandardScaler`, and `LabelEncoder`.
*   **Pandas & NumPy:** Data manipulation, numerical processing, and CSV parsing.

### Frontend
*   **HTML5 & CSS3:** Semantic markup and custom styling.
*   **Bootstrap 5.3:** Responsive grid system and professional UI components.
*   **JavaScript (Vanilla):** Client-side form state management and local-storage-based theme toggling.

---

## Environmental Parameters Evaluated

The model requires the following 7 inputs to generate a prediction:

1.  **Nitrogen (N):** 0 - 140 kg/ha
2.  **Phosphorus (P):** 5 - 145 kg/ha
3.  **Potassium (K):** 5 - 205 kg/ha
4.  **Temperature:** 8 - 44 °C
5.  **Humidity:** 14 - 100 %
6.  **Soil pH:** 3.5 - 10.0
7.  **Rainfall:** 20 - 300 mm

---

## Machine Learning Architecture

1.  **Data Preprocessing (`preprocess.py`):** Loads the dataset, drops null values, separates features from labels, and applies a `StandardScaler` to normalize numerical ranges.
2.  **Model Training (`train_model.py`):** The engine trains multiple candidate algorithms (Logistic Regression, KNN, Decision Tree, Random Forest) for performance comparison.
3.  **Bundle Generation:** The system automatically identifies the best performing model (currently **Random Forest** achieving **99.55%+ accuracy**) and serializes it along with the scaler and label encoder into a single `model/model.pkl` binary file for production use.

---

## Project Structure

```text
OptiCrop/
│
├── app.py                  # Main Flask application and route definitions
├── config.py               # Environment variable configurations
├── helper.py               # Form validation and data parsing logic
├── predictor.py            # Wrapper class for loading and querying the ML model
├── preprocess.py           # Data cleaning and scaling pipeline
├── train_model.py          # Algorithm training and accuracy evaluation
├── expand_dataset.py       # Script to inject new crops into the dataset
├── test.py                 # Smoke tests for Flask routes and model loading
├── requirements.txt        # Python package dependencies
├── Procfile                # Deployment configuration for PaaS providers
├── .gitignore              # Ignored files (excludes the large model.pkl)
├── README.md               # Project documentation
│
├── dataset/
│   └── crop_recommendation.csv  # 6,400+ row dataset
│
├── templates/
│   ├── home.html           # Landing page
│   ├── about.html          # Architecture overview
│   └── findyourcrop.html   # Main prediction dashboard
│
├── static/
|   ├── css/style.css       # Custom styling and theme variables
|    └── js/script.js        # Theme toggling and frontend logic
|
└── OptiCorp_Dashboard/
    ├── OptiCrop_About_Page.png
    ├── OptiCrop_Dashboard_Home_Page.png
    └── OptiCrop_Find_Your_Crop_Page.png

```

---

## Setup & Run Instructions

**Create and activate a virtual environment (recommended)**
   ```bash
   1. python -m venv venv
   venv\Scripts\activate

   2. Install dependencies
   pip install -r requirements.txt
   
   3. Build the extended dataset
   python expand_dataset.py
   
   4. Train the ML model (generates the model/model.pkl file)
   python train_model.py

   5. Run the application smoke tests
   python test.py

   6. Start the Flask server
   python app.py

```
---

## Dashboard Images
<img width="2880" height="1710" alt="OptiCorp_Dashboard_Home_Page" src="https://github.com/user-attachments/assets/342bd34a-20e5-4499-ae37-bc8e9ad70f43" />
<img width="2880" height="1788" alt="OptiCrop_About_Page" src="https://github.com/user-attachments/assets/89ac4d84-93c0-40a6-9ee3-ef8958913b36" />
<img width="2880" height="1800" alt="OptiCrop_Find_Your_Crop_Page" src="https://github.com/user-attachments/assets/168ad138-21cd-4289-9174-6eb6829222e8" />


