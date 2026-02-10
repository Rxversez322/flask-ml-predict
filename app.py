from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("heart_nb_model.pkl")

@app.route("/", methods=["GET", "POST"])
def predict():
    result = None

    if request.method == "POST":
        data = [
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]

        prediction = model.predict([data])[0]

        result = "มีความเสี่ยงโรคหัวใจ ❤️" if prediction == 1 else "ไม่มีความเสี่ยง 😊"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
