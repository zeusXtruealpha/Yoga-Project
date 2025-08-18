from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import torch
from torchvision import models, transforms
from tensorflow.keras.models import load_model
import time # Import time for logging
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ====== Preprocessing & Image Quality Checks ======

FIXED_SIZE = (256, 256)

def get_timestamp():
    """Helper function to get a formatted timestamp for logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', img_rgb)
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def image_to_base64_png(img):
    """Convert OpenCV image to base64 string as PNG (supports transparency)."""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Encode to PNG
    _, buffer = cv2.imencode('.png', img_rgb)
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def save_debug_images(processed_images, timestamp):
    """Save processed images locally for debugging."""
    try:
        import os
        debug_dir = "debug_images"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        for step, img_base64 in processed_images.items():
            if img_base64:
                # Decode base64
                img_data = base64.b64decode(img_base64)
                # Save as file
                filename = f"{debug_dir}/{timestamp}_{step}.jpg"
                if step == 'background_removed_transparent':
                    filename = f"{debug_dir}/{timestamp}_{step}.png"
                with open(filename, 'wb') as f:
                    f.write(img_data)
                print(f"    [DEBUG] Saved {filename}")
    except Exception as e:
        print(f"    [WARNING] Failed to save debug images: {e}")

def resize_image(img):
    print(f"    [DEBUG {get_timestamp()}] Resizing image from {img.shape[:2]} to {FIXED_SIZE}")
    return cv2.resize(img, FIXED_SIZE)

def check_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"    [DEBUG {get_timestamp()}] Noise check (Laplacian Var): {laplacian_var:.2f}")
    if laplacian_var < 50:
        return "High noise detected"
    return "Noise levels acceptable"

def check_white_pixels(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = np.sum(gray > 240) / gray.size
    print(f"    [DEBUG {get_timestamp()}] White pixel ratio check: {white_ratio:.2f}")
    if white_ratio > 0.4:
        return "Too many white/bright pixels (overexposed)"
    return "White pixel ratio acceptable"

def check_lighting(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    print(f"    [DEBUG {get_timestamp()}] Lighting check (Mean Intensity): {mean_intensity:.2f}")
    if mean_intensity < 50:
        return "Image too dark"
    elif mean_intensity > 200:
        return "Image too bright"
    return "Lighting conditions acceptable"

def check_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"    [DEBUG {get_timestamp()}] Blur check (Laplacian Var): {fm:.2f}")
    if fm < 100:  # threshold tuneable
        return "Image is blurry"
    return "Sharpness acceptable"

# --- NEW CHECKS ADDED HERE ---
def check_contrast(img):
    """Checks if the image has sufficient contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    print(f"    [DEBUG {get_timestamp()}] Contrast check (Std Dev): {contrast:.2f}")
    if contrast < 30:
        return f"Image has very low contrast"
    return "Contrast level acceptable"

def reduce_noise(img):
    """Reduces noise using Gaussian blur."""
    print(f"    [DEBUG {get_timestamp()}] Applying noise reduction...")
    # Apply Gaussian blur to reduce noise
    denoised = cv2.GaussianBlur(img, (3, 3), 0)
    return denoised

def remove_background_grabcut(img):
    """Removes background using GrabCut algorithm with transparent output."""
    print(f"    [DEBUG {get_timestamp()}] Applying background removal with GrabCut...")
    
    # Keep original image
    original = img.copy()
    
    # Initialize mask
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Models for GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Define rectangle for subject (adjust if needed)
    height, width = img.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    
    # Apply GrabCut algorithm
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # Prepare final mask
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = original * binary_mask[:, :, np.newaxis]
    
    # Convert to RGBA for transparent output
    output_rgba = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    output_rgba[:, :, 3] = binary_mask * 255
    
    # Convert back to BGR for consistency with other processing steps
    # We'll use a white background for the final output
    white_bg = np.ones_like(img) * 255
    final_result = white_bg * (1 - binary_mask[:, :, np.newaxis]) + result
    
    return final_result

def check_aspect_ratio(img):
    """Checks for extreme aspect ratios that would distort on resize."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        print(f"    [WARN {get_timestamp()}] Invalid image dimensions received: {h}x{w}")
        return "Invalid image dimensions"
    aspect_ratio = w / h
    print(f"    [DEBUG {get_timestamp()}] Aspect ratio check: {aspect_ratio:.2f}")
    if aspect_ratio > 3.0 or aspect_ratio < (1/3.0):
        return f"Warning: Extreme aspect ratio may cause distortion"
    return "Aspect ratio acceptable"
# --- END OF NEW CHECKS ---


def preprocess_and_validate(img):
    print(f"  [INFO {get_timestamp()}] Starting preprocessing and validation...")
    results = []
    processed_images = {}
    step_details = []

    # Store original image
    processed_images['original'] = image_to_base64(img)
    step_details.append({
        'step': 'Original Image',
        'description': 'Uploaded image as-is',
        'image_key': 'original'
    })

    # New: Aspect Ratio check on the original image
    aspect_result = check_aspect_ratio(img)
    results.append(aspect_result)
    step_details.append({
        'step': 'Aspect Ratio Check',
        'description': aspect_result,
        'image_key': 'original'
    })

    # Step 1: Resize
    img_resized = resize_image(img)
    processed_images['resized'] = image_to_base64(img_resized)
    results.append("Resized to fixed size")
    step_details.append({
        'step': 'Image Resizing',
        'description': 'Resized to 256x256 pixels for consistent processing',
        'image_key': 'resized'
    })

    # Step 2: Noise reduction (NEW)
    img_denoised = reduce_noise(img_resized)
    processed_images['denoised'] = image_to_base64(img_denoised)
    results.append("Noise reduction applied")
    step_details.append({
        'step': 'Noise Reduction',
        'description': 'Applied Gaussian blur (3x3) to reduce image noise',
        'image_key': 'denoised'
    })

    # Step 3: Background removal with GrabCut (NEW)
    img_no_bg = remove_background_grabcut(img_denoised)
    processed_images['background_removed'] = image_to_base64(img_no_bg)
    results.append("Background removed using GrabCut")
    step_details.append({
        'step': 'Background Removal (White)',
        'description': 'Removed background using GrabCut algorithm with white background',
        'image_key': 'background_removed'
    })
    
    # Also create transparent version for better visualization
    # Get the transparent version from the function
    original = img_denoised.copy()
    mask = np.zeros(img_denoised.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width = img_denoised.shape[:2]
    rect = (10, 10, width - 20, height - 20)
    cv2.grabCut(img_denoised, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    transparent_result = original * binary_mask[:, :, np.newaxis]
    output_rgba = cv2.cvtColor(transparent_result, cv2.COLOR_BGR2BGRA)
    output_rgba[:, :, 3] = binary_mask * 255
    processed_images['background_removed_transparent'] = image_to_base64_png(output_rgba)
    step_details.append({
        'step': 'Background Removal (Transparent)',
        'description': 'Removed background using GrabCut algorithm with transparent background',
        'image_key': 'background_removed_transparent'
    })

    # Step 4: Quality checks on processed image
    noise_result = check_noise(img_no_bg)
    results.append(noise_result)
    step_details.append({
        'step': 'Noise Level Check',
        'description': noise_result,
        'image_key': 'background_removed'
    })

    white_pixel_result = check_white_pixels(img_no_bg)
    results.append(white_pixel_result)
    step_details.append({
        'step': 'White Pixel Check',
        'description': white_pixel_result,
        'image_key': 'background_removed'
    })

    lighting_result = check_lighting(img_no_bg)
    results.append(lighting_result)
    step_details.append({
        'step': 'Lighting Check',
        'description': lighting_result,
        'image_key': 'background_removed'
    })

    blur_result = check_blur(img_no_bg)
    results.append(blur_result)
    step_details.append({
        'step': 'Blur Detection',
        'description': blur_result,
        'image_key': 'background_removed'
    })

    contrast_result = check_contrast(img_no_bg)
    results.append(contrast_result)
    step_details.append({
        'step': 'Contrast Check',
        'description': contrast_result,
        'image_key': 'background_removed'
    })

    print(f"  [INFO {get_timestamp()}] Preprocessing and validation complete.")
    
    # Save debug images
    timestamp = get_timestamp().replace(':', '-').replace(' ', '_')
    save_debug_images(processed_images, timestamp)
    
    return img_no_bg, results, processed_images, step_details
#=====================================================

# ====== Health Check ======
@app.route('/health', methods=['GET'])
def health():
    print(f"[INFO {get_timestamp()}] Health check endpoint was hit.")
    return jsonify({"status": "ok"}), 200

# ====== Load Emotion Model =====
MODEL_PATH = "../vit_model_epoch30.keras"
print(f"[SETUP {get_timestamp()}] Loading emotion model from: {MODEL_PATH}")

# Initialize model as None
model = None
class_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

try:
    if os.path.exists(MODEL_PATH):
        # Try to load the model with custom objects
        from tensorflow.keras.utils import custom_object_scope
        # You might need to import your custom layers here
        # For now, we'll handle the error gracefully
        model = load_model(MODEL_PATH)
        print(f"[SETUP {get_timestamp()}] Emotion model loaded successfully")
    else:
        print(f"[WARNING {get_timestamp()}] Emotion model file not found at: {MODEL_PATH}")
        print(f"[WARNING {get_timestamp()}] Running in demo mode with mock predictions")
except Exception as e:
    print(f"[WARNING {get_timestamp()}] Failed to load emotion model: {str(e)}")
    print(f"[WARNING {get_timestamp()}] Running in demo mode with mock predictions")

print(f"[SETUP {get_timestamp()}] Emotion Class labels: {class_labels}")

# ====== Load DeepLab Human Segmentation Model =====
print(f"[SETUP {get_timestamp()}] Loading DeepLabV3 model for human detection...")
deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
print(f"[SETUP {get_timestamp()}] DeepLabV3 model loaded successfully")

# ====== Utility for Emotion Prediction =====
def predict_emotion(img):
    print(f"  [INFO {get_timestamp()}] Starting emotion prediction...")
    
    if model is None:
        print(f"    [DEBUG {get_timestamp()}] Using mock prediction (model not loaded)")
        # Return a mock prediction for demo purposes
        import random
        mock_emotions = ["happy", "neutral", "sad"]
        predicted_label = random.choice(mock_emotions)
        print(f"  [INFO {get_timestamp()}] Mock emotion prediction complete. Result: {predicted_label}")
        return predicted_label
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    print(f"    [DEBUG {get_timestamp()}] Emotion model input tensor shape: {input_tensor.shape}")
    
    preds = model.predict(input_tensor)
    print(f"    [DEBUG {get_timestamp()}] Raw emotion predictions: {preds[0]}")
    
    class_idx = np.argmax(preds[0])
    predicted_label = class_labels[class_idx]
    print(f"  [INFO {get_timestamp()}] Emotion prediction complete. Result: {predicted_label}")
    return predicted_label

# ====== Utilities for Human Detection =====
def preprocess_for_deeplab(img):
    trf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return trf(img).unsqueeze(0)

def get_person_mask(output):
    output_predictions = output['out'].squeeze(0).argmax(0).byte().cpu().numpy()
    person_mask = np.where(output_predictions == 15, 255, 0).astype(np.uint8)
    return person_mask

def get_alignment_feedback(bbox, img_shape):
    x, y, w, h = bbox
    img_h, img_w = img_shape[:2]
    center_x = x + w // 2

    # Calculate percentages for debugging
    width_percentage = (w / img_w) * 100
    center_percentage = (center_x / img_w) * 100
    
    print(f"    [DEBUG {get_timestamp()}] BBox analysis: width={w}/{img_w} ({width_percentage:.1f}%), center_x={center_x}/{img_w} ({center_percentage:.1f}%)")

    feedback = []
    
    # Adjusted thresholds for better detection
    if w > img_w * 0.8:  # Changed from 0.6 to 0.8 (80%)
        feedback.append("Too close to camera")
    elif w < img_w * 0.15:  # Changed from 0.2 to 0.15 (15%)
        feedback.append("Too far from camera")
    
    if center_x < img_w * 0.25:  # Changed from 0.3 to 0.25 (25%)
        feedback.append("Move right")
    elif center_x > img_w * 0.75:  # Changed from 0.7 to 0.75 (75%)
        feedback.append("Move left")

    if not feedback:
        feedback.append("Aligned properly")

    print(f"    [DEBUG {get_timestamp()}] Alignment feedback: {feedback}")
    return feedback

def detect_person_and_feedback(img):
    print(f"  [INFO {get_timestamp()}] Starting person detection...")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_for_deeplab(img_rgb)
    print(f"    [DEBUG {get_timestamp()}] DeepLab input tensor shape: {input_tensor.shape}")

    with torch.no_grad():
        output = deeplab(input_tensor)

    mask = get_person_mask(output)
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        print(f"    [DEBUG {get_timestamp()}] Largest contour BBox: [x={x}, y={y}, w={w}, h={h}]")
        print(f"    [DEBUG {get_timestamp()}] Image shape: {img.shape}")
        print(f"    [DEBUG {get_timestamp()}] Contour area: {cv2.contourArea(largest)}")
        feedback = get_alignment_feedback((x, y, w, h), img.shape)
        print(f"  [INFO {get_timestamp()}] Person detected. Feedback generated.")
        return True, feedback
    else:
        print(f"  [INFO {get_timestamp()}] No person detected.")
        return False, ["No person detected"]

# ====== Flask Routes ======

@app.route('/predict', methods=['POST'])
def predict():
    print(f"\n\n===== [REQUEST {get_timestamp()}] /predict Endpoint Hit =====")
    if 'file' not in request.files:
        print(f"  [ERROR {get_timestamp()}] No file provided in request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print(f"  [INFO {get_timestamp()}] Received file: {file.filename}")
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"  [ERROR {get_timestamp()}] Failed to decode image.")
        return jsonify({'error': 'Invalid image'}), 400
    
    print(f"  [INFO {get_timestamp()}] Image decoded successfully. Original shape: {img.shape}")

    # Run pre-checks and preprocessing
    img_processed, precheck_feedback, processed_images, step_details = preprocess_and_validate(img)

    # Log pre-checks in terminal
    print("\n  --- Pre-check Results ---")
    for msg in precheck_feedback:
        print(f"    - {msg}")
    print("  -------------------------\n")


    # Predict emotion
    mood = predict_emotion(img_processed)
    print(f"  [RESULT {get_timestamp()}] Final Predicted Mood: {mood}")
    print(f"===== [REQUEST END {get_timestamp()}] /predict =====\n")

    return jsonify({
        'mood': mood,
        'precheck_feedback': precheck_feedback,
        'processed_images': processed_images,
        'step_details': step_details
    })


@app.route('/check_person', methods=['POST'])
def check_person():
    print(f"\n\n===== [REQUEST {get_timestamp()}] /check_person Endpoint Hit =====")
    if 'file' not in request.files:
        print(f"  [ERROR {get_timestamp()}] No file provided in request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print(f"  [INFO {get_timestamp()}] Received file: {file.filename}")
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"  [ERROR {get_timestamp()}] Failed to decode image.")
        return jsonify({'error': 'Invalid image'}), 400
    
    print(f"  [INFO {get_timestamp()}] Image decoded successfully. Original shape: {img.shape}")

    # Run pre-checks and preprocessing
    img_processed, precheck_feedback, processed_images, step_details = preprocess_and_validate(img)

    # Human detection - use original image for better alignment feedback
    person_present, feedback = detect_person_and_feedback(img)

    # Log in terminal
    print("\n  --- Pre-check Results ---")
    for msg in precheck_feedback:
        print(f"    - {msg}")
    print("  -------------------------\n")


    if person_present:
        print(f"  [RESULT {get_timestamp()}] Person detected: YES")
    else:
        print(f"  [RESULT {get_timestamp()}] Person detected: NO")

    print("\n  --- Alignment Feedback ---")
    for msg in feedback:
        print(f"    - {msg}")
    print("  --------------------------\n")

    print(f"===== [REQUEST END {get_timestamp()}] /check_person =====\n")

    return jsonify({
        'person_detected': person_present,
        'feedback': feedback,
        'precheck_feedback': precheck_feedback,
        'processed_images': processed_images,
        'step_details': step_details
    })


@app.route('/preprocess', methods=['POST'])
def preprocess_only():
    """Endpoint to get only the preprocessing steps without prediction."""
    print(f"\n\n===== [REQUEST {get_timestamp()}] /preprocess Endpoint Hit =====")
    if 'file' not in request.files:
        print(f"  [ERROR {get_timestamp()}] No file provided in request.")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    print(f"  [INFO {get_timestamp()}] Received file: {file.filename}")
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        print(f"  [ERROR {get_timestamp()}] Failed to decode image.")
        return jsonify({'error': 'Invalid image'}), 400
    
    print(f"  [INFO {get_timestamp()}] Image decoded successfully. Original shape: {img.shape}")

    # Run preprocessing only
    img_processed, precheck_feedback, processed_images, step_details = preprocess_and_validate(img)

    # Log in terminal
    print("\n  --- Preprocessing Steps ---")
    for i, step in enumerate(step_details, 1):
        print(f"    {i}. {step['step']}: {step['description']}")
    print("  ---------------------------\n")

    print(f"===== [REQUEST END {get_timestamp()}] /preprocess =====\n")

    return jsonify({
        'precheck_feedback': precheck_feedback,
        'processed_images': processed_images,
        'step_details': step_details
    })


if __name__ == '__main__':
    print(f"[SETUP {get_timestamp()}] Starting Flask server...")
    # Using debug=False for production, but True is fine for development
    app.run(host='0.0.0.0', port=5000, debug=True)