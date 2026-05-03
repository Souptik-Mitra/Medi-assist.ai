import json
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types

from symptom_inference import features_from_text

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BACKEND_ROOT, ".env"), override=True)

FRONTEND_DIR = os.path.abspath(os.path.join(BACKEND_ROOT, "..", "frontend"))

app = Flask(__name__)
CORS(app)

_PLACEHOLDER_KEYS = frozenset(
    {"", "your_gemini_api_key_here", "paste_your_key_here"}
)


def clear_dead_local_proxy_settings():
    dead_proxy_values = {"http://127.0.0.1:9", "https://127.0.0.1:9"}
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        value = (os.getenv(key) or "").strip().lower()
        if value in dead_proxy_values:
            os.environ.pop(key, None)
            os.environ.pop(key.lower(), None)


clear_dead_local_proxy_settings()
api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
client = None
if api_key and api_key not in _PLACEHOLDER_KEYS:
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini API client initialized (using cloud model for /api/diagnose).")
    except Exception as e:
        print(f"Gemini client failed to initialize: {e}. Falling back to local ML when needed.")
else:
    print(
        "No valid GEMINI_API_KEY in backend/.env — using local ML for /api/diagnose."
    )

MODEL_PATH = os.path.join(BACKEND_ROOT, "models", "symptom_model.pkl")
local_bundle = None
if os.path.exists(MODEL_PATH):
    try:
        loaded = joblib.load(MODEL_PATH)
        if isinstance(loaded, dict) and "pipeline" in loaded:
            local_bundle = loaded
            print("Local ML bundle loaded (structured symptoms model).")
        else:
            print(
                "Obsolete symptom_model.pkl format. Run: python train_model.py"
            )
    except Exception as e:
        print(f"Error loading local ML model: {e}")


def predict_local_ml(age, gender, symptoms_text: str):
    if not local_bundle:
        return None
    pipe = local_bundle["pipeline"]
    rf = getattr(pipe, "named_steps", {}).get("rf")
    if rf is not None and hasattr(rf, "n_jobs"):
        # Keep inference single-threaded to avoid Windows IPC failures in restricted environments.
        rf.n_jobs = 1
    medians = local_bundle["medians"]
    mode_gender = local_bundle.get("mode_gender", "Female")
    cols = local_bundle["feature_cols"]
    row = features_from_text(symptoms_text, age, gender, medians, mode_gender)
    X = pd.DataFrame([row])[cols]
    return pipe.predict(X)[0]


def local_fallback_response(label):
    if label is not None:
        return {
            "possible_conditions": [f"(Local ML) {label}"],
            "remedies_and_advice": [
                "Rest and monitor your symptoms.",
                "This prediction comes from a model trained on synthetic tabular data mapped from your text; it is not a clinical diagnosis.",
            ],
            "doctor_recommendation": "Always consult a qualified healthcare provider for medical advice.",
            "disclaimer": "This tool is not a substitute for medical advice. Please consult a licensed doctor.",
        }

    return {
        "possible_conditions": [
            "(Mock) Common Cold",
            "(Mock) Seasonal Allergies",
        ],
        "remedies_and_advice": [
            "Rest and drink plenty of fluids.",
            "Simulated response: no API key and no trained local model. Run python train_model.py in the backend folder.",
        ],
        "doctor_recommendation": "Always consult a real doctor for medical advice.",
        "disclaimer": "This tool is not a substitute for medical advice. Please consult a licensed doctor.",
    }


SYSTEM_INSTRUCTION = """
You are Medi-Assist.AI, an educational health and symptom analysis assistant.
Your goal is to interpret user symptoms, provide possible (non-diagnostic) conditions, suggest safe home or OTC remedies if appropriate, and ALWAYS recommend consulting a doctor.

CRITICAL RULES:
1. NEVER provide a definitive medical diagnosis.
2. ALWAYS include a disclaimer that this is for educational purposes and not a substitute for professional medical advice.
3. Keep the tone empathetic, professional, and clear.
4. Output your response STRICTLY as a JSON object with the following schema:
{
  "possible_conditions": ["condition 1", "condition 2"],
  "remedies_and_advice": ["advice 1", "advice 2"],
  "doctor_recommendation": "A strong statement advising them to see a healthcare professional.",
  "disclaimer": "This tool is not a substitute for medical advice. Please consult a licensed doctor."
}
"""


@app.route("/api/diagnose", methods=["POST"])
def diagnose():
    data = request.json
    if not data or not data.get("symptoms"):
        return jsonify({"error": "Symptoms are required."}), 400

    symptoms = data.get("symptoms")
    age = data.get("age", "Not provided")
    gender = data.get("gender", "Not provided")

    if not client:
        label = predict_local_ml(age, gender, symptoms)
        return jsonify(local_fallback_response(label))

    user_prompt = f"Patient Profile: Age: {age}, Gender: {gender}. Symptoms reported: {symptoms}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
            ),
        )

        try:
            result_json = json.loads(response.text)
            return jsonify(result_json)
        except json.JSONDecodeError:
            return (
                jsonify(
                    {
                        "error": "Failed to parse AI response. Raw output: "
                        + response.text
                    }
                ),
                500,
            )

    except Exception as e:
        print(f"Gemini request failed: {e}. Falling back to local ML.")
        label = predict_local_ml(age, gender, symptoms)
        response = local_fallback_response(label)
        response["llm_error"] = str(e)
        return jsonify(response)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def serve_frontend_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def serve_frontend_assets(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
