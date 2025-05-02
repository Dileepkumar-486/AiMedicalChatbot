import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ✅ Define class labels (Modify as needed)
class_names = ['Acne', 'ChickenPox', 'DarkSpots', 'Eczema', 'Healthy', 
               'Moles', 'Monkeypox', 'NailFungus', 'Psoriasis', 'Puffy Eyes', 'Rash', 
               'RingWorm', 'Scabies', 'Scars', 'SkinCancer', 'SunDamage', 'Vitiligo', 
               'Warts', 'Wrinkles']

# ✅ Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Define image transformations for **inference** (No randomness)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure correct size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Load the trained model
def load_model(model_path, num_classes=len(class_names)):
    """Loads EfficientNet model and updates classifier for correct number of classes."""
    model = models.efficientnet_b0(pretrained=False)  # Load base model
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)  # Adjust classifier

    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully!")
    return model

# ✅ Prediction function with confidence threshold
def predict_with_threshold(model, image_path, transform, class_names, threshold=80):
    """Performs image classification with confidence filtering."""
    
    # 🔹 Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    processed_image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # 🔹 Perform inference
    with torch.no_grad():
        outputs = model(processed_image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
        top_probs, top_classes = torch.topk(probabilities, 3)  # Get top-3 predictions

    # 🔹 Convert results
    top_probs = top_probs.squeeze().cpu().numpy() * 100  # Convert to percentages
    top_classes = top_classes.squeeze().cpu().numpy()  # Convert tensor to array

    # 🔥 Check confidence level
    if top_probs[0] >= threshold:
        return {
            "Prediction": class_names[top_classes[0]], 
            "Confidence": f"{top_probs[0]:.2f}%"
        }
    else:
        return {
            "Warning": "Low confidence. Possible classes:",
            "Options": [
                {"Class": class_names[top_classes[i]], "Confidence": f"{top_probs[i]:.2f}%"}
                for i in range(len(top_classes))
            ]
        }

# ✅ Run Prediction
model_path = r"D:\\projects\\chat\\models\\woww.pth"  # 🔹 Change to your trained model path
image_path = r"D:\\projects\\chat\\acne.jpeg"  # 🔹 Change to your test image path

# Load model and make prediction
model = load_model(model_path)
result = predict_with_threshold(model, image_path, transform, class_names, threshold=65)

# ✅ Print result
print(result)
