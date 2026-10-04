"""
Evaluates the Production Vision Service on the Real-World Validation Suite (46 images).

Outputs structured results and saves evaluation report to evaluation/real_world_report.json.
"""

import os
import sys
import json
import time
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from services.vision_service import VisionService

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "evaluation", "real_world_manifest.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "evaluation", "real_world_report.json")


def run_real_world_evaluation():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found: {MANIFEST_PATH}. Run scripts/build_real_world_validation.py first.")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print("=" * 75)
    print("Running Real-World Validation Benchmark Suite (46 Images)")
    print("=" * 75)

    vision = VisionService()

    total_images = len(manifest)
    correct_overall = 0
    food_total = 0
    food_correct = 0
    non_food_total = 0
    non_food_rejected = 0
    false_food_claims = 0
    unknown_total = 0
    unknown_handled = 0
    latencies = []
    results = []

    for idx, item in enumerate(manifest):
        img_path = item["absolute_path"]
        exp_class = item["expected_class"]
        exp_type = item["expected_type"]
        cat = item["category"]
        fname = item["filename"]

        if not os.path.exists(img_path):
            print(f"[{idx+1:02d}/{total_images:02d}] Missing image: {img_path}")
            continue

        res = vision.analyze_image(img_path)
        latencies.append(res["latency_ms"])

        pred_status = res["status"]
        pred_name = res["name"]
        pred_cat = res["category"]
        conf = res["confidence"]
        desc = res["visual_description"]

        is_correct = False

        if exp_type == "non_food":
            non_food_total += 1
            if pred_status == "not_food":
                non_food_rejected += 1
                is_correct = True
            else:
                false_food_claims += 1
        elif exp_class == "unknown_food":
            unknown_total += 1
            # For OOD/Unknown food, correct behavior is either status="uncertain" or a non-forced description
            if pred_status == "uncertain":
                unknown_handled += 1
                is_correct = True
            elif pred_status == "not_food":
                # Some exotic things might be gate rejected or recognized
                is_correct = False
            else:
                is_correct = False
        else:
            # Known food / fruit / dish
            food_total += 1
            if pred_status == "recognized" and pred_name == exp_class:
                food_correct += 1
                is_correct = True
            elif pred_status == "uncertain" and exp_class in ["thali_mixed", "rice_dal_combo"]:
                # Composite dishes marked uncertain with composite description are also valid safe responses
                food_correct += 1
                is_correct = True

        if is_correct:
            correct_overall += 1
            tag = "[PASS]"
        else:
            tag = "[FAIL]"

        name_str = pred_name if pred_name else ("not_food" if pred_status == "not_food" else "uncertain/desc")
        print(f"{tag} {cat:<15} | {fname:<22} | Exp: {exp_class:<15} -> Pred: {name_str:<15} ({conf:.2f}, {pred_status})")

        results.append({
            "filename": fname,
            "category": cat,
            "expected_class": exp_class,
            "expected_type": exp_type,
            "pred_status": pred_status,
            "pred_name": pred_name,
            "pred_category": pred_cat,
            "confidence": conf,
            "visual_description": desc,
            "is_correct": is_correct,
            "latency_ms": res["latency_ms"],
            "raw_predictions": res.get("raw_predictions", [])
        })

    overall_acc = (correct_overall / total_images) * 100 if total_images > 0 else 0
    food_acc = (food_correct / food_total) * 100 if food_total > 0 else 0
    rejection_rate = (non_food_rejected / non_food_total) * 100 if non_food_total > 0 else 0
    false_food_rate = (false_food_claims / non_food_total) * 100 if non_food_total > 0 else 0
    unknown_rate = (unknown_handled / unknown_total) * 100 if unknown_total > 0 else 0
    avg_lat = float(np.mean(latencies)) if latencies else 0

    print("=" * 75)
    print("REAL-WORLD VALIDATION SUMMARY")
    print("=" * 75)
    print(f"Total Test Images:               {total_images}")
    print(f"Overall Accuracy:                 {overall_acc:.1f}% ({correct_overall}/{total_images})")
    print(f"Food/Fruit Recognition Accuracy:  {food_acc:.1f}% ({food_correct}/{food_total})")
    print(f"Non-Food Rejection Rate (15 obj): {rejection_rate:.1f}% ({non_food_rejected}/{non_food_total})")
    print(f"False Food Rate on Non-Food:      {false_food_rate:.1f}% ({false_food_claims}/{non_food_total})")
    print(f"Unknown / OOD Safe Handling:      {unknown_rate:.1f}% ({unknown_handled}/{unknown_total})")
    print(f"Average CPU Inference Latency:    {avg_lat:.1f} ms")
    print("=" * 75)

    report = {
        "model_name": "Production Hierarchical Gated SigLIP Vision Service",
        "total_images": total_images,
        "overall_accuracy_pct": round(overall_acc, 1),
        "overall_correct": correct_overall,
        "food_recognition_accuracy_pct": round(food_acc, 1),
        "food_correct": food_correct,
        "food_total": food_total,
        "non_food_rejection_rate_pct": round(rejection_rate, 1),
        "non_food_rejected": non_food_rejected,
        "non_food_total": non_food_total,
        "false_food_rate_pct": round(false_food_rate, 1),
        "unknown_handling_rate_pct": round(unknown_rate, 1),
        "unknown_handled": unknown_handled,
        "unknown_total": unknown_total,
        "avg_latency_ms": round(avg_lat, 1),
        "detailed_results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Detailed real-world report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run_real_world_evaluation()
