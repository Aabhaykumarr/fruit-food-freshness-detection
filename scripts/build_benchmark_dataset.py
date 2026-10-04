"""
Builds a curated, reproducible benchmark evaluation dataset across 5 required categories:
- fruits
- foods
- non_food
- difficult
- unknown
"""

import os
import json
import urllib.request
import cv2
from PIL import Image

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation")

# Curated high-quality, authentic benchmark images
BENCHMARK_SPEC = [
    # 1. FRUITS
    {
        "category": "fruits",
        "expected_class": "apple",
        "expected_type": "fruit",
        "filename": "apple_01.jpg",
        "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=600&auto=format&fit=crop",
        "description": "Fresh red delicious apple on white background"
    },
    {
        "category": "fruits",
        "expected_class": "banana",
        "expected_type": "fruit",
        "filename": "banana_01.jpg",
        "url": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600&auto=format&fit=crop",
        "description": "Ripe yellow bananas in bunch"
    },
    {
        "category": "fruits",
        "expected_class": "orange",
        "expected_type": "fruit",
        "filename": "orange_01.jpg",
        "url": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=600&auto=format&fit=crop",
        "description": "Whole fresh ripe orange with citrus skin"
    },
    {
        "category": "fruits",
        "expected_class": "mango",
        "expected_type": "fruit",
        "filename": "mango_01.jpg",
        "url": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=600&auto=format&fit=crop",
        "description": "Ripe yellow mango on plain background"
    },
    {
        "category": "fruits",
        "expected_class": "strawberry",
        "expected_type": "fruit",
        "filename": "strawberry_01.jpg",
        "url": "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?w=600&auto=format&fit=crop",
        "description": "Fresh whole strawberries"
    },
    {
        "category": "fruits",
        "expected_class": "watermelon",
        "expected_type": "fruit",
        "filename": "watermelon_01.jpg",
        "url": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=600&auto=format&fit=crop",
        "description": "Fresh sliced watermelon piece"
    },
    {
        "category": "fruits",
        "expected_class": "grapes",
        "expected_type": "fruit",
        "filename": "grapes_01.jpg",
        "url": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=600&auto=format&fit=crop",
        "description": "Bunch of fresh purple grapes"
    },
    {
        "category": "fruits",
        "expected_class": "pomegranate",
        "expected_type": "fruit",
        "filename": "pomegranate_01.jpg",
        "url": "https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=600&auto=format&fit=crop",
        "description": "Fresh whole and cut pomegranate with arils"
    },

    # 2. FOODS
    {
        "category": "foods",
        "expected_class": "pizza",
        "expected_type": "food",
        "filename": "pizza_01.jpg",
        "url": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600&auto=format&fit=crop",
        "description": "Whole cheese and tomato pizza on wooden board"
    },
    {
        "category": "foods",
        "expected_class": "burger",
        "expected_type": "food",
        "filename": "burger_01.jpg",
        "url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop",
        "description": "Classic beef/cheese burger with sesame bun"
    },
    {
        "category": "foods",
        "expected_class": "pasta",
        "expected_type": "food",
        "filename": "pasta_01.jpg",
        "url": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=600&auto=format&fit=crop",
        "description": "Italian pasta with tomato basil sauce"
    },
    {
        "category": "foods",
        "expected_class": "sandwich",
        "expected_type": "food",
        "filename": "sandwich_01.jpg",
        "url": "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=600&auto=format&fit=crop",
        "description": "Toasted club sandwich cut into halves"
    },
    {
        "category": "foods",
        "expected_class": "samosa",
        "expected_type": "food",
        "filename": "samosa_01.jpg",
        "url": "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop",
        "description": "Crispy golden fried Indian samosas with chutney"
    },
    {
        "category": "foods",
        "expected_class": "biryani",
        "expected_type": "food",
        "filename": "biryani_01.jpg",
        "url": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop",
        "description": "Spiced chicken biryani rice dish in handi"
    },
    {
        "category": "foods",
        "expected_class": "dosa",
        "expected_type": "food",
        "filename": "dosa_01.jpg",
        "url": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop",
        "description": "Crispy South Indian masala dosa with chutney and sambar"
    },
    {
        "category": "foods",
        "expected_class": "idli",
        "expected_type": "food",
        "filename": "idli_01.jpg",
        "url": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=600&auto=format&fit=crop",
        "description": "Steamed white idlis with coconut chutney"
    },
    {
        "category": "foods",
        "expected_class": "paneer",
        "expected_type": "food",
        "filename": "paneer_01.jpg",
        "url": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=600&auto=format&fit=crop",
        "description": "Paneer butter masala / paneer curry in bowl"
    },
    {
        "category": "foods",
        "expected_class": "roti",
        "expected_type": "food",
        "filename": "roti_01.jpg",
        "url": "https://images.unsplash.com/photo-1626074353765-517a681e40be?w=600&auto=format&fit=crop",
        "description": "Indian flatbread chapati / roti stack"
    },

    # 3. NON-FOOD (Crucial Negative Samples)
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "phone_01.jpg",
        "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600&auto=format&fit=crop",
        "description": "Modern smartphone on dark surface"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "laptop_01.jpg",
        "url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&auto=format&fit=crop",
        "description": "Laptop computer open on wooden desk"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "circuit_arduino_01.jpg",
        "url": "https://images.unsplash.com/photo-1518770660439-4636190af475?w=600&auto=format&fit=crop",
        "description": "Electronic circuit board / microcontroller"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "person_face_01.jpg",
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=600&auto=format&fit=crop",
        "description": "Human face portrait portrait photo"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "chair_01.jpg",
        "url": "https://images.unsplash.com/photo-1592078615290-033ee584e267?w=600&auto=format&fit=crop",
        "description": "Wooden chair in minimalist room"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "book_01.jpg",
        "url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=600&auto=format&fit=crop",
        "description": "Hardcover open book on table"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "bottle_01.jpg",
        "url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=600&auto=format&fit=crop",
        "description": "Reusable stainless steel water bottle"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "watch_01.jpg",
        "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop",
        "description": "Analog wristwatch on white background"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "car_01.jpg",
        "url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=600&auto=format&fit=crop",
        "description": "Sports car vehicle on road"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "cat_01.jpg",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600&auto=format&fit=crop",
        "description": "Domestic cat portrait looking at camera"
    },

    # 4. DIFFICULT / COMPLEX FOOD SAMPLES
    {
        "category": "difficult",
        "expected_class": "dal",
        "expected_type": "food",
        "filename": "dal_tadka_01.jpg",
        "url": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=600&auto=format&fit=crop",
        "description": "Yellow dal tadka / lentil curry in steel bowl with garnish"
    },
    {
        "category": "difficult",
        "expected_class": "curry",
        "expected_type": "food",
        "filename": "mixed_curry_01.jpg",
        "url": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&auto=format&fit=crop",
        "description": "Rich dark Indian curry dish in kadai"
    },
    {
        "category": "difficult",
        "expected_class": "apple",
        "expected_type": "fruit",
        "filename": "apple_shadow_01.jpg",
        "url": "https://images.unsplash.com/photo-1570913149827-d2ac84ab3f9a?w=600&auto=format&fit=crop",
        "description": "Apple in dramatic shadow and moody ambient lighting"
    },
    {
        "category": "difficult",
        "expected_class": "banana",
        "expected_type": "fruit",
        "filename": "banana_partial_01.jpg",
        "url": "https://images.unsplash.com/photo-1528825871115-3581a5387919?w=600&auto=format&fit=crop",
        "description": "Peeled and sliced banana on dark background"
    },
    {
        "category": "difficult",
        "expected_class": "thali_mixed",
        "expected_type": "food",
        "filename": "indian_thali_01.jpg",
        "url": "https://images.unsplash.com/photo-1610057099443-fde8c4d50f91?w=600&auto=format&fit=crop",
        "description": "Complete Indian thali with multiple small bowls of curry, rice, and roti"
    },

    # 5. UNKNOWN / OUT-OF-DISTRIBUTION FOODS (Intentionally uncommon/exotic dishes)
    {
        "category": "unknown",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "rambutan_01.jpg",
        "url": "https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=600&auto=format&fit=crop",
        "description": "Exotic tropical hairy rambutan fruit"
    },
    {
        "category": "unknown",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "durian_01.jpg",
        "url": "https://images.unsplash.com/photo-1546548970-71785318a17b?w=600&auto=format&fit=crop",
        "description": "Exotic spiky durian fruit cut open"
    },
    {
        "category": "unknown",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "paella_01.jpg",
        "url": "https://images.unsplash.com/photo-1534080564583-6be75777b70a?w=600&auto=format&fit=crop",
        "description": "Spanish seafood paella rice dish"
    },
    {
        "category": "unknown",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "dimsum_01.jpg",
        "url": "https://images.unsplash.com/photo-1541696432-82c6da8ce7bf?w=600&auto=format&fit=crop",
        "description": "Chinese steamed dim sum dumplings in bamboo steamer"
    }
]

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    manifest = []

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    print(f"Downloading and curating {len(BENCHMARK_SPEC)} benchmark images...")

    for idx, item in enumerate(BENCHMARK_SPEC, 1):
        cat_dir = os.path.join(EVAL_DIR, item["category"])
        os.makedirs(cat_dir, exist_ok=True)
        file_path = os.path.join(cat_dir, item["filename"])

        print(f"[{idx}/{len(BENCHMARK_SPEC)}] {item['category']}/{item['filename']} ... ", end="", flush=True)

        # Download if not present
        if not os.path.exists(file_path):
            try:
                req = urllib.request.Request(item["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = resp.read()
                with open(file_path, "wb") as f:
                    f.write(data)

                # Verify valid image with PIL / OpenCV
                img = Image.open(file_path)
                img.verify()
                print("Downloaded & verified")
            except Exception as e:
                print(f"FAILED: {e}")
                continue
        else:
            print("Already exists")

        manifest.append({
            "relative_path": os.path.join("evaluation", item["category"], item["filename"]).replace("\\", "/"),
            "absolute_path": os.path.abspath(file_path),
            "category": item["category"],
            "expected_class": item["expected_class"],
            "expected_type": item["expected_type"],
            "description": item["description"]
        })

    manifest_path = os.path.join(EVAL_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBenchmark dataset ready! Total verified images: {len(manifest)}")
    print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    main()
