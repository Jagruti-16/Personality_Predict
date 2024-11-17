from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load("personality_model.joblib")

personality_mapping = {
    0: "The Supervisor (ESTJ)",
    1: "The Commander (ENTJ)",
    2: "The Provider (ESFJ)",
    3: "The Giver (ENFJ)",
    4: "The Inspector (ISTJ)",
    5: "The Nurturer (ISFJ)",
    6: "The Mastermind (INTJ)",
    7: "The Counselor (INFJ)",
    8: "The Doer (ESTP)",
    9: "The Performer (ESFP)",
    10: "The Visionary (ENTP)",
    11: "The Champion (ENFP)",
    12: "The Craftsman (ISTP)",
    13: "The Composer (ISFP)",
    14: "The Thinker (INTP)",
    15: "The Idealist (INFP)"
}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request (JSON)
        data = request.json
        responses = data.get('responses')

        # Convert responses to numpy array (model expects this format)
        responses = np.array(responses).reshape(1, -1)

        # Use the model to predict personality type
        prediction = model.predict(responses)[0]

        # Get the personality type from the mapping
        personality_type = personality_mapping.get(prediction, "Unknown")

        # Return the result as a JSON response
        return jsonify({"personality_type": personality_type}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)