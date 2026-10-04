"""
Evaluation of the Final Integrated Hierarchical Vision Service on the 37-Image Benchmark Dataset.

Compares:
1. Baseline: YOLO-World v2 Small
2. Final Production Service: Hierarchical Gated SigLIP Vision Pipeline (VisionService)
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vision_service import VisionService
from scripts.benchmark_models import EQUIVALENTS, is_class_match, MANIFEST_PATH

RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "final_evaluation_report.json")


def evaluate_production_vision_service():
    print("=" * 70)
    print("Evaluating Production Hierarchical Vision Service on 37-Image Benchmark")
    print("=" * 70)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    vision_service = VisionService()

    results = []
    latencies = []

    for idx, item in enumerate(manifest):
        img_path = item["absolute_path"]
        category = item["category"]
        expected_class = item["expected_class"]
        filename = item.get("filename", os.path.basename(item["relative_path"]))

        t0 = time.perf_counter()
        prediction = vision_service.analyze_image(img_path)
        latency = (time.perf_counter() - t0) * 1000
        latencies.append(latency)

        status = prediction.get("status")
        pred_name = prediction.get("name")
        conf = prediction.get("confidence", 0.0)

        if status == "not_food":
            pred_class = "non_food"
        elif status == "uncertain":
            pred_class = "unknown_food"
        else:
            pred_class = pred_name

        match = is_class_match(pred_class, expected_class)
        # For unknown items, status == "uncertain" or "unknown_food" is a success
        if category == "unknown" and (status == "uncertain" or pred_class == "unknown_food"):
            match = True

        results.append({
            "filename": filename,
            "category": category,
            "expected_class": expected_class,
            "status": status,
            "pred_name": pred_name,
            "pred_class": pred_class,
            "confidence": conf,
            "visual_description": prediction.get("visual_description", ""),
            "is_correct": match,
            "latency_ms": latency,
        })

        tag = "PASS" if match else "FAIL"
        print(f"[{idx+1:02d}/37] [{tag:4s}] {category:10s} | {filename:24s} | Exp: {expected_class:15s} -> Pred: {str(pred_class):16s} ({conf:.2f}, {status})")

    # Compute Metrics
    df = pd.DataFrame(results)
    total = len(df)
    overall_correct = int(df["is_correct"].sum())
    overall_acc = overall_correct / total * 100

    # Food/Fruit Recognition Accuracy (fruits + foods + difficult)
    food_df = df[df["category"].isin(["fruits", "foods", "difficult"])]
    food_total = len(food_df)
    food_correct = int(food_df["is_correct"].sum())
    food_acc = food_correct / food_total * 100 if food_total > 0 else 0.0

    # Non-food Rejection Rate
    non_food_df = df[df["category"] == "non_food"]
    non_food_total = len(non_food_df)
    non_food_rejected = int((non_food_df["status"] == "not_food").sum())
    non_food_rej_rate = non_food_rejected / non_food_total * 100 if non_food_total > 0 else 0.0

    # False food rate on non-food
    false_food_on_non_food = non_food_total - non_food_rejected
    false_food_rate = false_food_on_non_food / non_food_total * 100 if non_food_total > 0 else 0.0

    # False food name rate on food images
    false_name_on_food = food_total - food_correct
    false_food_name_rate = false_name_on_food / food_total * 100 if food_total > 0 else 0.0

    # Unknown handling
    unk_df = df[df["category"] == "unknown"]
    unk_total = len(unk_df)
    unk_handled = int(unk_df["is_correct"].sum())
    unk_rate = unk_handled / unk_total * 100 if unk_total > 0 else 0.0

    avg_latency = float(np.mean(latencies))

    summary = {
        "model_name": "Hierarchical Gated SigLIP Vision Service",
        "total_images": total,
        "overall_accuracy_pct": round(overall_acc, 1),
        "overall_correct": overall_correct,
        "food_recognition_accuracy_pct": round(food_acc, 1),
        "food_correct": food_correct,
        "food_total": food_total,
        "non_food_rejection_rate_pct": round(non_food_rej_rate, 1),
        "non_food_rejected": non_food_rejected,
        "non_food_total": non_food_total,
        "false_food_rate_pct": round(false_food_rate, 1),
        "false_food_name_rate_pct": round(false_food_name_rate, 1),
        "unknown_handling_rate_pct": round(unk_rate, 1),
        "unknown_handled": unk_handled,
        "unknown_total": unk_total,
        "avg_latency_ms": round(avg_latency, 1),
        "detailed_results": results
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Overall Accuracy:                 {overall_acc:.1f}% ({overall_correct}/{total})")
    print(f"Food/Fruit Recognition Accuracy:  {food_acc:.1f}% ({food_correct}/{food_total})")
    print(f"Non-Food Rejection Rate:          {non_food_rej_rate:.1f}% ({non_food_rejected}/{non_food_total})")
    print(f"False Food Rate on Non-Food:      {false_food_rate:.1f}% ({false_food_on_non_food}/{non_food_total})")
    print(f"False Food Name Rate on Food:     {false_food_name_rate:.1f}% ({false_name_on_food}/{food_total})")
    print(f"Unknown / OOD Handling Rate:      {unk_rate:.1f}% ({unk_handled}/{unk_total})")
    print(f"Average CPU Inference Latency:    {avg_latency:.1f} ms")
    print("=" * 70)
    print(f"Detailed evaluation report saved to: {RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    evaluate_production_vision_service()
