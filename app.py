from flask import Flask, render_template, request, jsonify
from utils import model_predict # apo to arxio utils (utils.py pou briskete sto idio katalogo),
                                # kano import in function me onoma model_predict
# einai kali praktiki na exo mia boithitiko arxio (utils) to opoio exei oles tis xrisimes functions
# pou xrisimopoio sto main arxio (Java2)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('content') # pernoume to periexomeno tou email
    prediction = model_predict(email)
    return render_template("index.html", prediction=prediction, email=email)

# Create an API endpoint
@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)  # Get data posted as a json
    email = data['content']
    prediction = model_predict(email)
    return jsonify({'prediction': prediction, 'email': email})  # Return prediction

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
