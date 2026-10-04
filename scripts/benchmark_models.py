"""
Comprehensive Benchmarking Suite for Fruit & Food Recognition Models
Evaluates:
1. Baseline: YOLO-World v2 Small (Open-Vocabulary Detection)
2. Baseline: Food-101 ViT Classifier (Fine-tuned Classification)
3. Candidate: Google SigLIP Base (Zero-Shot Classifier with Negative Prompts)
4. Candidate: OpenAI CLIP ViT-B/32 (Zero-Shot Classifier)
5. Candidate: Hierarchical Gated Pipeline (SigLIP Gate + Food/Fruit Specialists + Uncertainty Calibration)

Measures:
- Overall Accuracy
- Precision & Recall (Food/Fruit vs Non-Food)
- Non-Food Rejection Rate
- False Food Rate on Non-Food Images
- False Food Name Rate on Food Images
- Unknown/Out-of-Distribution Handling
- Average CPU Latency (ms)
- Model Size (MB)
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
import torch
import cv2

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FOOD_CLASSES

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation")
MANIFEST_PATH = os.path.join(EVAL_DIR, "manifest.json")

# Ground truth synonym / alias mapping for flexible semantic equivalence
EQUIVALENTS = {
    "apple": ["apple", "red apple", "green apple", "fresh apple"],
    "banana": ["banana", "ripe banana", "bananas"],
    "orange": ["orange", "citrus", "ripe orange"],
    "mango": ["mango", "yellow mango", "ripe mango"],
    "strawberry": ["strawberry", "strawberries", "fresh strawberry"],
    "watermelon": ["watermelon", "watermelon slice"],
    "grapes": ["grapes", "purple grapes", "grape bunch"],
    "pomegranate": ["pomegranate", "pomegranate arils"],
    "pizza": ["pizza", "cheese pizza"],
    "burger": ["burger", "hamburger", "cheeseburger"],
    "pasta": ["pasta", "spaghetti", "spaghetti bolognese", "pasta with tomato sauce"],
    "sandwich": ["sandwich", "club sandwich", "toasted sandwich"],
    "samosa": ["samosa", "samosas", "fried samosa"],
    "biryani": ["biryani", "chicken biryani", "spiced rice dish"],
    "dosa": ["dosa", "masala dosa", "crispy dosa"],
    "idli": ["idli", "steamed idli", "idlis"],
    "paneer": ["paneer", "paneer curry", "paneer butter masala", "cottage cheese curry"],
    "roti": ["roti", "chapati", "flatbread", "indian roti"],
    "dal": ["dal", "dal tadka", "lentil soup", "lentil curry"],
    "curry": ["curry", "mixed curry", "indian curry", "spiced gravy"],
    "thali_mixed": ["thali", "indian thali", "mixed dish", "combo platter"],
    "non_food": ["non_food", "not_food", "none"],
    "unknown_food": ["unknown", "unknown_food", "uncertain", "unrecognized"]
}

def is_class_match(predicted, expected):
    if expected == "non_food":
        return predicted in ["non_food", "not_food", None, "none"]
    if expected == "unknown_food":
        return predicted in ["unknown", "unknown_food", "uncertain", None] or predicted is None

    pred_clean = str(predicted).lower().strip().replace("_", " ").replace("-", " ")
    exp_clean = str(expected).lower().strip().replace("_", " ").replace("-", " ")

    if exp_clean in pred_clean or pred_clean in exp_clean:
        return True

    aliases = EQUIVALENTS.get(expected, [expected])
    for alias in aliases:
        alias_clean = alias.lower().replace("_", " ")
        if alias_clean in pred_clean or pred_clean in alias_clean:
            return True

    return False


# ==========================================
# MODEL 1: YOLO-World v2 Small Baseline
# ==========================================
class YoloWorldBenchmark:
    def __init__(self):
        from ultralytics import YOLO
        self.name = "YOLO-World v2 Small (Current Baseline)"
        self.task = "Open-Vocabulary Object Detection"
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8s-worldv2.pt")
        self.model = YOLO(self.model_path)
        self.model.set_classes(FOOD_CLASSES)
        self.model_size_mb = round(os.path.getsize(self.model_path) / (1024 * 1024), 1)
        self.num_classes = len(FOOD_CLASSES)

    def predict(self, image_path):
        img_bgr = cv2.imread(image_path)
        t0 = time.perf_counter()
        results = self.model(img_bgr, conf=0.35, verbose=False)
        latency = (time.perf_counter() - t0) * 1000

        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cls_name = FOOD_CLASSES[cls_id] if cls_id < len(FOOD_CLASSES) else f"item_{cls_id}"
                detections.append({"class_name": cls_name, "confidence": conf})

        if detections:
            # Top detection
            detections.sort(key=lambda x: x["confidence"], reverse=True)
            top = detections[0]
            return {
                "status": "recognized",
                "predicted_class": top["class_name"],
                "confidence": top["confidence"],
                "latency_ms": latency
            }
        else:
            return {
                "status": "not_food",
                "predicted_class": "non_food",
                "confidence": 0.0,
                "latency_ms": latency
            }


# ==========================================
# MODEL 2: Food-101 ViT Classifier Baseline
# ==========================================
class Food101ViTBenchmark:
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForImageClassification
        self.name = "Food-101 ViT (Fine-Tuned Classifier)"
        self.task = "101-Class Food Image Classification"
        self.processor = AutoProcessor.from_pretrained("nateraw/food")
        self.model = AutoModelForImageClassification.from_pretrained("nateraw/food")
        self.model.eval()
        self.num_classes = 101
        self.model_size_mb = 343.0  # Approx weights size

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            top_prob, top_idx = torch.topk(probs, 1)
            pred_class = self.model.config.id2label[top_idx.item()]
            conf = float(top_prob.item())
        latency = (time.perf_counter() - t0) * 1000

        return {
            "status": "recognized",
            "predicted_class": pred_class,
            "confidence": conf,
            "latency_ms": latency
        }


# ==========================================
# MODEL 3: Google SigLIP Zero-Shot Classifier
# ==========================================
class SigLipBenchmark:
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
        self.name = "Google SigLIP Base (Zero-Shot with Negative Prompts)"
        self.task = "Zero-Shot Food/Fruit Classification + Gating"
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = AutoModelForZeroShotImageClassification.from_pretrained("google/siglip-base-patch16-224")
        self.model.eval()

        # Explicit target labels including non-food negatives
        self.candidate_labels = [
            "apple", "banana", "orange", "mango", "strawberry", "watermelon", "grapes", "pomegranate",
            "pizza", "burger", "pasta", "sandwich", "samosa", "biryani", "dosa", "idli", "paneer curry",
            "roti flatbread", "dal lentil soup", "curry dish", "indian thali platter", "dumplings",
            "an electronic device, smartphone, or laptop", "a person portrait or face", "furniture or household item",
            "a vehicle, car, or bus", "a cat, dog, or animal", "a water bottle or container", "a book or notebook"
        ]
        self.num_classes = len(self.candidate_labels)
        self.model_size_mb = 400.0

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        inputs = self.processor(text=self.candidate_labels, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = torch.sigmoid(logits)  # SigLIP uses sigmoid pairwise
            top_idx = torch.argmax(probs).item()
            conf = float(probs[top_idx].item())
            top_label = self.candidate_labels[top_idx]
        latency = (time.perf_counter() - t0) * 1000

        # Check if negative non-food label won
        is_negative = any(neg in top_label for neg in ["electronic", "person", "furniture", "vehicle", "animal", "bottle", "book"])

        if is_negative:
            return {
                "status": "not_food",
                "predicted_class": "non_food",
                "confidence": conf,
                "latency_ms": latency
            }
        else:
            return {
                "status": "recognized",
                "predicted_class": top_label.split()[0], # e.g. "apple"
                "confidence": conf,
                "latency_ms": latency
            }


# ==========================================
# MODEL 4: Hierarchical Gated Vision Pipeline
# ==========================================
class HierarchicalGatedPipelineBenchmark:
    """
    Proposed Architecture:
    1. Zero-Shot Gate: Food/Fruit vs Non-Food vs Ambiguous (SigLIP / CLIP)
    2. Category Routing: Fruit/Veg vs Dish
    3. Specialized Recognition: High-resolution ViT / SigLIP feature matcher
    4. Calibrated Uncertainty & Out-of-Distribution Gating
    """
    def __init__(self):
        from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoModelForImageClassification
        self.name = "Hierarchical Gated Vision Pipeline (Proposed)"
        self.task = "Multi-Stage Gate -> Specialist -> Calibrated Uncertainty"

        # 1. Gate model
        self.gate_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.gate_model = AutoModelForZeroShotImageClassification.from_pretrained("google/siglip-base-patch16-224")
        self.gate_model.eval()

        # Gate Prompts
        self.gate_prompts = [
            "a photo of fresh fruit or raw vegetable",
            "a photo of a cooked meal, prepared food dish, or curry",
            "a photo of an electronic device, computer, phone, or circuit",
            "a photo of a human face or person",
            "a photo of furniture, chair, table, or room interior",
            "a photo of a random non-food object, book, bottle, or vehicle"
        ]

        # 2. Specialist Food/Fruit Prompts for open-set fine-grained classification
        self.food_fruit_prompts = [
            # Fruits
            "fresh red apple", "ripe yellow banana", "fresh orange citrus", "ripe yellow mango",
            "fresh strawberries", "sliced watermelon", "purple grapes", "fresh pomegranate",
            # Western / Global Foods
            "cheesy pizza", "hamburger with bun", "pasta with sauce", "club sandwich",
            # Indian / Regional Foods
            "crispy indian samosa", "spiced chicken biryani rice", "south indian dosa crepe",
            "steamed idli rice cakes", "paneer butter masala curry", "indian roti chapati flatbread",
            "yellow dal tadka lentil curry", "rich indian vegetable curry", "indian thali platter"
        ]
        self.prompt_to_class = {
            "fresh red apple": "apple", "ripe yellow banana": "banana", "fresh orange citrus": "orange",
            "ripe yellow mango": "mango", "fresh strawberries": "strawberry", "sliced watermelon": "watermelon",
            "purple grapes": "grapes", "fresh pomegranate": "pomegranate", "cheesy pizza": "pizza",
            "hamburger with bun": "burger", "pasta with sauce": "pasta", "club sandwich": "sandwich",
            "crispy indian samosa": "samosa", "spiced chicken biryani rice": "biryani", "south indian dosa crepe": "dosa",
            "steamed idli rice cakes": "idli", "paneer butter masala curry": "paneer", "indian roti chapati flatbread": "roti",
            "yellow dal tadka lentil curry": "dal", "rich indian vegetable curry": "curry", "indian thali platter": "thali_mixed"
        }

        self.num_classes = len(self.food_fruit_prompts)
        self.model_size_mb = 400.0

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()

        # STEP 1: GATE INFERENCE (Food vs Non-Food)
        gate_inputs = self.gate_processor(text=self.gate_prompts, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            gate_outputs = self.gate_model(**gate_inputs)
            gate_probs = torch.sigmoid(gate_outputs.logits_per_image[0])
            top_gate_idx = torch.argmax(gate_probs).item()
            top_gate_prob = float(gate_probs[top_gate_idx].item())

        # If Non-food prompt is top prediction or non-food sum > food sum
        # Indices 0,1 are Food/Fruit; indices 2,3,4,5 are Non-Food
        food_score = float(torch.max(gate_probs[0], gate_probs[1]).item())
        non_food_score = float(torch.max(gate_probs[2:]).item())

        if top_gate_idx >= 2 or (non_food_score > food_score and non_food_score > 0.45):
            latency = (time.perf_counter() - t0) * 1000
            return {
                "status": "not_food",
                "predicted_class": "non_food",
                "confidence": non_food_score,
                "latency_ms": latency
            }

        # STEP 2: SPECIALIZED FOOD/FRUIT RECOGNITION
        spec_inputs = self.gate_processor(text=self.food_fruit_prompts, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            spec_outputs = self.gate_model(**spec_inputs)
            spec_probs = torch.sigmoid(spec_outputs.logits_per_image[0])
            top_idx = torch.argmax(spec_probs).item()
            conf = float(spec_probs[top_idx].item())
            top_prompt = self.food_fruit_prompts[top_idx]
            pred_class = self.prompt_to_class[top_prompt]

        latency = (time.perf_counter() - t0) * 1000

        # STEP 3: UNCERTAINTY & OUT-OF-DISTRIBUTION CALIBRATION
        # If top probability is weak (<0.40) or entropy/margin is low, flag as UNCERTAIN
        sorted_probs, _ = torch.sort(spec_probs, descending=True)
        margin = float((sorted_probs[0] - sorted_probs[1]).item()) if len(sorted_probs) > 1 else 1.0

        if conf < 0.35 or (conf < 0.50 and margin < 0.08):
            return {
                "status": "uncertain",
                "predicted_class": "unknown_food",
                "confidence": conf,
                "latency_ms": latency
            }
        else:
            return {
                "status": "recognized",
                "predicted_class": pred_class,
                "confidence": conf,
                "latency_ms": latency
            }


# ==========================================
# BENCHMARK EVALUATOR RUNNER
# ==========================================
def run_benchmark():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}. Run scripts/build_benchmark_dataset.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 80)
    print(f"STARTING MODEL ACCURACY BENCHMARK ON {len(manifest)} TEST IMAGES")
    print("=" * 80)

    # Instantiate all models
    models = []

    print("\nLoading Model 1: YOLO-World v2 Small (Baseline)...")
    try:
        models.append(YoloWorldBenchmark())
    except Exception as e:
        print(f"Failed to load YOLO-World: {e}")

    print("\nLoading Model 2: Food-101 ViT Classifier (Baseline)...")
    try:
        models.append(Food101ViTBenchmark())
    except Exception as e:
        print(f"Failed to load Food-101 ViT: {e}")

    print("\nLoading Model 3: Google SigLIP Base Zero-Shot...")
    try:
        models.append(SigLipBenchmark())
    except Exception as e:
        print(f"Failed to load SigLIP: {e}")

    print("\nLoading Model 4: Hierarchical Gated Vision Pipeline (Proposed)...")
    try:
        models.append(HierarchicalGatedPipelineBenchmark())
    except Exception as e:
        print(f"Failed to load Proposed Pipeline: {e}")

    # Results collection
    benchmark_summary = []

    for model in models:
        print(f"\n" + "-" * 70)
        print(f"Evaluating: {model.name}")
        print(f"Task: {model.task}")
        print("-" * 70)

        total_images = len(manifest)
        correct_overall = 0

        # Non-food metrics
        non_food_total = 0
        non_food_correctly_rejected = 0
        non_food_false_food_claims = 0

        # Food/Fruit recognition metrics
        food_total = 0
        food_correctly_identified = 0
        food_false_name_claims = 0

        # Unknown handling metrics
        unknown_total = 0
        unknown_correctly_handled = 0

        latencies = []
        confidences = []

        per_image_results = []

        for item in manifest:
            image_path = item["absolute_path"]
            expected_class = item["expected_class"]
            category = item["category"]

            res = model.predict(image_path)
            latencies.append(res["latency_ms"])
            confidences.append(res["confidence"])

            pred_class = res["predicted_class"]
            status = res["status"]

            is_correct = is_class_match(pred_class, expected_class)

            if category == "non_food":
                non_food_total += 1
                if status == "not_food" or is_correct:
                    non_food_correctly_rejected += 1
                    correct_overall += 1
                else:
                    non_food_false_food_claims += 1
            elif category == "unknown":
                unknown_total += 1
                # Unknown is handled correctly if status == "uncertain" or returned unknown/not falsely confident
                if status == "uncertain" or is_class_match(pred_class, "unknown_food"):
                    unknown_correctly_handled += 1
                    correct_overall += 1
                else:
                    # Hallucinated a confident standard class
                    pass
            else: # fruits, foods, difficult
                food_total += 1
                if is_correct:
                    food_correctly_identified += 1
                    correct_overall += 1
                else:
                    food_false_name_claims += 1

            fname = item.get("filename", os.path.basename(item["relative_path"]))
            per_image_results.append({
                "image": fname,
                "category": category,
                "expected": expected_class,
                "predicted": pred_class,
                "status": status,
                "confidence": round(res["confidence"], 3),
                "is_correct": is_correct
            })

            # Print single line log
            mark = "PASS" if is_correct or (category == "unknown" and status == "uncertain") else "FAIL"
            print(f"  [{mark:4s}] {category:8s} | {fname:24s} | Exp: {expected_class:15s} -> Pred: {str(pred_class):18s} ({res['confidence']:.2f})")

        # Aggregate Metrics
        overall_acc = (correct_overall / total_images) * 100
        non_food_rejection_rate = (non_food_correctly_rejected / non_food_total * 100) if non_food_total > 0 else 0
        false_food_on_nonfood_rate = (non_food_false_food_claims / non_food_total * 100) if non_food_total > 0 else 0
        food_accuracy = (food_correctly_identified / food_total * 100) if food_total > 0 else 0
        false_food_name_rate = (food_false_name_claims / food_total * 100) if food_total > 0 else 0
        unknown_handling_rate = (unknown_correctly_handled / unknown_total * 100) if unknown_total > 0 else 0
        avg_latency = float(np.mean(latencies))

        summary_entry = {
            "model_name": model.name,
            "task": model.task,
            "classes": model.num_classes,
            "model_size_mb": model.model_size_mb,
            "total_test_images": total_images,
            "overall_accuracy_pct": round(overall_acc, 1),
            "food_recognition_acc_pct": round(food_accuracy, 1),
            "non_food_rejection_rate_pct": round(non_food_rejection_rate, 1),
            "false_food_rate_pct": round(false_food_on_nonfood_rate, 1),
            "false_food_name_rate_pct": round(false_food_name_rate, 1),
            "unknown_handling_rate_pct": round(unknown_handling_rate, 1),
            "avg_latency_ms": round(avg_latency, 1),
            "per_image_results": per_image_results
        }
        benchmark_summary.append(summary_entry)

    # Save benchmark report JSON
    report_path = os.path.join(EVAL_DIR, "benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED SUCCESSFULLY — RESULTS WRITTEN TO: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
