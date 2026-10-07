"""
End-to-end inference for Potato Storage Condition Detection: 
Stage 1 localizer -> crop -> Stage 2 classifier -> lot summary.
"""
import argparse
import json
from collections import Counter
import numpy as np
import cv2
import timm
import torch
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

EVAL_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_localizer(weights_path):
    return YOLO(weights_path)

def load_classifier(weights_path, model_name, class_names, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model

def run_pipeline(img, localizer, classifier, class_names, device, conf=0.25):
    """
    Runs the two-stage pipeline on the provided image.
    img: numpy array (RGB) or PIL Image or string path.
    """
    if isinstance(img, Image.Image):
        img = np.array(img.convert("RGB"))
    elif isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = localizer.predict(img, conf=conf, verbose=False)[0]
    
    counts = Counter()
    per_tuber_records = []
    
    # We will draw on a copy of the image
    annotated_image = img.copy()
    
    # Severity color mapping (RGB)
    # Healthy = Green, Sprouted = Yellow, Damaged = Orange, Defected = Orange-Red, Diseased = Red
    color_map = {
        "potato": (16, 185, 129),                # Healthy Green
        "sprouted potato": (245, 158, 11),       # Amber
        "damaged potato": (249, 115, 22),        # Orange
        "defected potato": (239, 68, 68),        # Red-Orange
        "diseased-fungal potato": (185, 28, 28)  # Deep Red
    }

    boxes = results.boxes.xyxy.cpu().numpy()
    localizer_confs = results.boxes.conf.cpu().numpy()

    for box, l_conf in zip(boxes, localizer_confs):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        tensor = EVAL_TF(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = classifier(tensor)
            probs = torch.nn.functional.softmax(preds, dim=1)[0]
            pred_idx = probs.argmax().item()
            c_conf = probs[pred_idx].item()
            
        pred_class = class_names[pred_idx]
        counts[pred_class] += 1
        
        per_tuber_records.append({
            "bbox": [x1, y1, x2, y2],
            "class": pred_class,
            "confidence": c_conf
        })
        
        # Draw bounding box and label
        color = color_map.get(pred_class.lower(), (255, 255, 255))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{pred_class} {c_conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_image, (x1, y1 - 25), (x1 + w, y1), color, -1)
        # Draw text
        cv2.putText(annotated_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated_image, per_tuber_records, dict(counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--localizer-weights", default="runs/detect/runs/localizer/potato_localizer-2/weights/best.pt")
    parser.add_argument("--classifier-weights", default="best_seresnet50.pt")
    parser.add_argument("--classifier-model", default="seresnet50")
    parser.add_argument("--class-names", default="data/class_names.json")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.class_names) as f:
        class_names = json.load(f)

    localizer = load_localizer(args.localizer_weights)
    classifier = load_classifier(args.classifier_weights, args.classifier_model, class_names, device)

    ann_img, records, counts = run_pipeline(
        args.image, localizer, classifier, class_names, device, args.conf
    )
    
    print(f"\nDetected {len(records)} tubers")
    for cls, n in counts.items():
        print(f"  {cls}: {n}")
