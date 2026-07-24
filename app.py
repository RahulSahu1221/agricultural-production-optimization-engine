from flask import Flask, render_template, request

from config import Config
from predictor import CropPredictor
from helper import parse_and_validate_form

app = Flask(__name__)
app.config.from_object(Config)

# Load the trained model bundle once at startup.
predictor = CropPredictor(model_path=Config.MODEL_PATH)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop", methods=["GET"])
def findyourcrop():
    return render_template("findyourcrop.html", model_ready=predictor.loaded)


@app.route("/predict", methods=["POST"])
def predict():
    if not predictor.loaded:
        return render_template(
            "findyourcrop.html",
            errors=["Model is not available. Please run train_model.py first."],
            model_ready=False,
        )

    features, errors, raw_values = parse_and_validate_form(request.form)

    if errors:
        return render_template(
            "findyourcrop.html",
            errors=errors,
            form_values=raw_values,
            model_ready=True,
        )

    try:
        result = predictor.predict(features)
    except Exception as exc:
        return render_template(
            "findyourcrop.html",
            errors=[f"Prediction failed: {exc}"],
            form_values=raw_values,
            model_ready=True,
        )

    return render_template(
        "findyourcrop.html",
        prediction=result,
        accuracy=predictor.get_accuracy_display(),
        model_used=predictor.model_name,
        form_values=raw_values,
        model_ready=True,
    )


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
