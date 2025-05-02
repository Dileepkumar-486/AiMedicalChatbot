import os
from flask import Flask, request, jsonify, session
from flask_cors import CORS  # type: ignore
import torch
from pymongo import MongoClient  # type: ignore
from transformers import BertTokenizer, BertForSequenceClassification
from huggingface_hub import login
import functionalities as fun
from stt import recognize_speech
from image_classify import load_model, predict_with_threshold, transform, class_names

# ✅ Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session management
CORS(app)

# ✅ Load Text-Based Medical Chatbot Model
HF_TOKEN = "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"
MODEL_PATH = r"D:\\projects\\chat\\bert_medical_classifier_pytorch"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

login(token=HF_TOKEN)

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
bert_model.eval()

# ✅ Load Image Classification Model
IMAGE_MODEL_PATH = r"D:\\projects\\chat\\models\\woww.pth"
image_model = load_model(IMAGE_MODEL_PATH)

# ✅ Create uploads folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Store ongoing diagnosis sessions globally
ongoing_diagnoses = {}


@app.route("/chat", methods=["POST"])
def chat():
    """Handles chatbot interactions with follow-up questions."""
    data = request.json
    user_input = data.get("message", "").strip().lower()
    user_id = data.get("user_id", "default_user")  # Track user conversations

    if not user_input:
        return jsonify({"error": "Empty input"}), 400
    
    # ✅ Handle greetings and goodbyes first
    response = fun.handle_greetings_and_goodbyes(user_input)
    if response:
        return jsonify({"response": response})

    # ✅ Ensure user history exists
    if user_id not in ongoing_diagnoses:
        ongoing_diagnoses[user_id] = {"last_query_type": None}

    last_query_type = ongoing_diagnoses[user_id].get("last_query_type")

    # ✅ Detect if user is in an ongoing diagnosis session
    if user_id in ongoing_diagnoses and "follow_up_questions" in ongoing_diagnoses[user_id]:
        diagnosis_data = ongoing_diagnoses[user_id]
        follow_up_index = diagnosis_data["follow_up_index"]
        follow_up_questions = diagnosis_data["follow_up_questions"]
        user_responses = diagnosis_data["user_responses"]

        # Store user response
        user_responses.append({
            "question": follow_up_questions[follow_up_index - 1],
            "answer": user_input
        })

        # If all questions are answered, generate final diagnosis
        if follow_up_index >= len(follow_up_questions):
            combined_responses = "\n".join([f"Q: {q['question']} | A: {q['answer']}" for q in user_responses])

            final_diagnosis = fun.generate_final_diagnosis(
                diagnosis_data["initial_input"], combined_responses
            )

            # ✅ Reset session to avoid topic mixing
            del ongoing_diagnoses[user_id]  # Clear session

            return jsonify({"response": final_diagnosis if final_diagnosis else "I couldn't determine a conclusive diagnosis."})

        # Ask the next follow-up question
        next_question = follow_up_questions[follow_up_index]
        diagnosis_data["follow_up_index"] += 1  
        return jsonify({"response": next_question})


    if not fun.classify_query(user_input):
        return jsonify({"response": "This is not a medical query. I can only assist with health-related questions."})
    
    # ✅ Classify the user query type
    query_type = fun.classify_query_type(user_input)

    # ✅ If user switches topics, reset session
    if last_query_type and last_query_type != query_type:
        del ongoing_diagnoses[user_id]  # Remove old session
        ongoing_diagnoses[user_id] = {"last_query_type": query_type}

    ongoing_diagnoses[user_id]["last_query_type"] = query_type  # Store current query type

    # ✅ Handle different query types
    if query_type == "unknown":
        return jsonify({"response": "I'm sorry, I didn't quite catch that. Can you give me more details or ask something else related to your health?"})

    elif query_type == "treatment":
        treatment = fun.generate_treatment(user_input)
        return jsonify({"response": treatment})

    elif query_type == "diagnosis":
        follow_up_questions = fun.generate_best_questions(user_input)

        if not follow_up_questions:
            return jsonify({"response": "I couldn't generate follow-up questions. Please describe your symptoms more clearly."})

        # ✅ Reset diagnosis session before starting a new one
        ongoing_diagnoses[user_id] = {
            "follow_up_questions": follow_up_questions,
            "user_responses": [],
            "initial_input": user_input,
            "follow_up_index": 1,
            "last_query_type": "diagnosis"
        }
        return jsonify({"response": follow_up_questions[0]})

    elif query_type == "general_medical":
        medical_info = fun.generate_general_medical_info(user_input)
        return jsonify({"response": medical_info})

    elif query_type == "medicine_info":
        medicine_info = fun.generate_medicine_info(user_input)
        return jsonify({"response": medicine_info})

    elif query_type == "health_advice":
        health_advice = fun.generate_health_advice(user_input)
        return jsonify({"response": health_advice})

    elif query_type == "emergency_advice":
        emergency_tips = fun.generate_emergency_advice(user_input)
        return jsonify({"response": emergency_tips})

    return jsonify({"response": "I'm a medical assistant and can only help with health-related questions."})


@app.route("/speech", methods=["POST"])
def speech_to_text():
    """Converts speech to text using STT model."""
    try:
        text = recognize_speech()
        return jsonify({"transcription": text})
    except Exception as e:
        return jsonify({"error": f"Speech recognition failed: {str(e)}"}), 500


from torchvision import transforms
from PIL import Image
import torch

SKIN_CLASSIFIER_MODEL = r"D:\projects\chat\models\image_classify_model.pth"
binary_model = load_model(SKIN_CLASSIFIER_MODEL, num_classes=2)  # Binary classification (2 classes)

# ✅ Load Skin Disease Classification Model (20 Classes)
DISEASE_CLASSIFIER_MODEL = r"D:\projects\chat\models\woww.pth"
image_model = load_model(DISEASE_CLASSIFIER_MODEL, num_classes=19)  # Multi-class classification (20 classes)

# ✅ Define Image Transform for Binary Classification
binary_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_skin_or_non_skin(image_path):
    """Predicts if an image is skin-related using the binary classifier."""
    image = Image.open(image_path).convert("RGB")
    image = binary_transform(image).unsqueeze(0)  # Apply transforms & add batch dimension

    with torch.no_grad():
        output = binary_model(image)  # This will return a tensor with 2 values

    probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()  # Get class index (0 or 1)

    return predicted_class == 1  # Assuming class 1 is "skin", class 0 is "non-skin"

@app.route("/predict-image", methods=["POST"])
def predict_image():
    """Handles image-based predictions, first checking if the image is skin-related."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # ✅ Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # ✅ Step 1: Check if image is skin-related using the binary classifier
        is_skin = predict_skin_or_non_skin(image_path)  # Fixed function

        if not is_skin:
            os.remove(image_path)
            return jsonify({"response": "This is not a skin-related image. Please upload a valid skin image."})

        # ✅ Step 2: If it's a skin image, classify the disease
        result = predict_with_threshold(image_model, image_path, transform, class_names, threshold=80)
        os.remove(image_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"Image prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)