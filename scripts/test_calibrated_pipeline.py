"""
Tests Calibrated Multi-Stage Hierarchical Gated Pipeline:
- Stage 1: Zero-Shot Non-Food vs Food Gate (SigLIP)
- Stage 2: Softmax-Normalized Specialized Category Classifier (Fruits / Indian Dishes / Global Foods)
- Stage 3: Calibrated Uncertainty / Out-of-Distribution Rejection
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from scripts.benchmark_models import EQUIVALENTS, is_class_match, MANIFEST_PATH

class CalibratedGatedPipeline:
    def __init__(self):
        self.name = "Calibrated Hierarchical Gated Pipeline"
        self.task = "Gate -> Normalized Specialist -> Calibrated Uncertainty"
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("google/siglip-base-patch16-224")
        self.model.eval()

        # Gate Prompts
        self.gate_prompts = [
            "a photo of fresh fruit or raw edible vegetable",
            "a photo of a cooked meal, prepared food dish, or curry",
            "a photo of an electronic device, computer, phone, or gadget",
            "a photo of a human person or human face",
            "a photo of furniture, chair, table, or room interior",
            "a photo of a non-food household object, book, bottle, vehicle, or animal"
        ]

        # Explicit Specialized Target Labels
        self.target_labels = [
            "apple", "banana", "orange", "mango", "strawberry", "watermelon", "grapes", "pomegranate",
            "pizza", "burger", "pasta", "sandwich", "samosa", "biryani", "dosa", "idli", "paneer curry",
            "roti flatbread", "dal lentil soup", "curry dish", "indian thali platter"
        ]
        self.label_to_class = {
            "apple": "apple", "banana": "banana", "orange": "orange", "mango": "mango",
            "strawberry": "strawberry", "watermelon": "watermelon", "grapes": "grapes", "pomegranate": "pomegranate",
            "pizza": "pizza", "burger": "burger", "pasta": "pasta", "sandwich": "sandwich",
            "samosa": "samosa", "biryani": "biryani", "dosa": "dosa", "idli": "idli",
            "paneer curry": "paneer", "roti flatbread": "roti", "dal lentil soup": "dal",
            "curry dish": "curry", "indian thali platter": "thali_mixed"
        }

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")

        # 1. GATE
        gate_inputs = self.processor(text=self.gate_prompts, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            gate_outputs = self.model(**gate_inputs)
            gate_logits = gate_outputs.logits_per_image[0]
            gate_probs = torch.softmax(gate_logits, dim=-1)

        food_score = float((gate_probs[0] + gate_probs[1]).item())
        non_food_score = float(torch.sum(gate_probs[2:]).item())

        if non_food_score > 0.55 or gate_probs[2] > 0.40 or gate_probs[3] > 0.40 or gate_probs[4] > 0.40 or gate_probs[5] > 0.40:
            return {
                "status": "not_food",
                "predicted_class": "non_food",
                "confidence": non_food_score
            }

        # 2. TARGET CLASSIFICATION
        inputs = self.processor(text=self.target_labels, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            # Softmax normalization for proper probability distribution
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            top_prob = float(sorted_probs[0].item())
            second_prob = float(sorted_probs[1].item())
            top_label = self.target_labels[sorted_indices[0].item()]
            pred_class = self.label_to_class[top_label]

        # 3. UNCERTAINTY CALIBRATION
        # Margin ratio test: if top probability is weak (<0.15 on 21 classes) or margin is near zero, mark UNCERTAIN
        margin = top_prob - second_prob

        if top_prob < 0.12 or (top_prob < 0.20 and margin < 0.04):
            return {
                "status": "uncertain",
                "predicted_class": "unknown_food",
                "confidence": top_prob,
                "visual_description": f"Visually resembles {pred_class} or related food, but confidence is low"
            }
        else:
            return {
                "status": "recognized",
                "predicted_class": pred_class,
                "confidence": top_prob
            }

def main():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    pipe = CalibratedGatedPipeline()
    print("Evaluating Calibrated Gated Pipeline...")

    total = len(manifest)
    correct = 0
    non_food_rej = 0
    non_food_total = 0
    unknown_handled = 0
    unknown_total = 0
    food_correct = 0
    food_total = 0

    for item in manifest:
        res = pipe.predict(item["absolute_path"])
        cat = item["category"]
        exp = item["expected_class"]
        pred = res["predicted_class"]
        stat = res["status"]

        is_match = is_class_match(pred, exp)

        if cat == "non_food":
            non_food_total += 1
            if stat == "not_food":
                non_food_rej += 1
                correct += 1
        elif cat == "unknown":
            unknown_total += 1
            if stat == "uncertain" or is_match:
                unknown_handled += 1
                correct += 1
        else:
            food_total += 1
            if is_match:
                food_correct += 1
                correct += 1

        mark = "PASS" if is_match or (cat == "unknown" and stat == "uncertain") else "FAIL"
        fname = os.path.basename(item["relative_path"])
        print(f"  [{mark:4s}] {cat:8s} | {fname:24s} | Exp: {exp:15s} -> Pred: {str(pred):18s} ({res['confidence']:.2f}, {stat})")

    print("\n" + "=" * 60)
    print(f"Overall Accuracy:           {correct / total * 100:.1f}% ({correct}/{total})")
    print(f"Food/Fruit Recognition Acc: {food_correct / food_total * 100:.1f}% ({food_correct}/{food_total})")
    print(f"Non-Food Rejection Rate:    {non_food_rej / non_food_total * 100:.1f}% ({non_food_rej}/{non_food_total})")
    print(f"Unknown Handling Rate:      {unknown_handled / unknown_total * 100:.1f}% ({unknown_handled}/{unknown_total})")
    print("=" * 60)

if __name__ == "__main__":
    main()
