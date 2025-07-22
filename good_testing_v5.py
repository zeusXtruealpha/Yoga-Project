import cv2
import torch
import numpy as np
from torchvision import models, transforms

deeplab = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

def preprocess(img):
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

    feedback = []
    if w > img_w * 0.6:
        feedback.append("Too close to camera")
    elif w < img_w * 0.2:
        feedback.append("Too far from camera")
    
    if center_x < img_w * 0.3:
        feedback.append("Move right")
    elif center_x > img_w * 0.7:
        feedback.append("Move left")

    if not feedback:
        feedback.append("Aligned properly")

    return feedback

def segment_and_check_alignment(image_path):
    original = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    input_tensor = preprocess(image_rgb)

    with torch.no_grad():
        output = deeplab(input_tensor)

    mask = get_person_mask(output)
    mask_resized = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Get bounding box from mask
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        feedback = get_alignment_feedback((x, y, w, h), original.shape)

        # Show feedback on image
        annotated = original.copy()
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for i, msg in enumerate(feedback):
            cv2.putText(annotated, msg, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Alignment Check", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No person detected")

# === Example ===
segment_and_check_alignment("test.jpg")
