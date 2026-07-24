# 🌱 OptiCrop

## Smart Agricultural Production Optimization Engine

OptiCrop is a Machine Learning based crop recommendation system that predicts the most suitable crop using soil and environmental parameters.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Pandas
- NumPy
- Scikit-Learn

## Project Structure

```
OptiCrop/
│
├── app.py
├── train_model.py
├── requirements.txt
├── dataset/
├── model/
├── templates/
├── static/
└── README.md
```

## Input Parameters

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

## Output

Recommended Crop

## Future Scope

- Cloud Deployment
- IoT Integration
- Real-Time Weather API
- Mobile Application

## Model

The dataset (`dataset/crop_recommendation.csv`) contains 2,200 samples across
22 crop classes. `train_model.py` trains and compares four algorithms
(Logistic Regression, KNN, Decision Tree, Random Forest) and automatically
saves the best performer — a **Random Forest** achieving **99.55% test
accuracy** — to `model/model.pkl`, bundled together with its `StandardScaler`
and `LabelEncoder`.

## Setup & Run

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Re)train the model — writes model/model.pkl
python train_model.py

# 4. Run the tests
python test.py

# 5. Start the app
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Deployment

A `Procfile` (`web: python app.py`) is included for platforms like Heroku or
Render. Set `SECRET_KEY`, `DEBUG`, `HOST`, and `PORT` as environment
variables in production instead of relying on `.env` (which is gitignored).
