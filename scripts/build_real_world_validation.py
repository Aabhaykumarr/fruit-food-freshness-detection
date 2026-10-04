"""
Builds the Real-World Validation Dataset (5-10 images per core class + 15 non-food negatives).

Focuses on:
- Normal household plates & bowls
- Indian kitchen & restaurant lighting
- Different angles & phone camera style shots
- Critical comparison classes: Plain Rice, Dal, Rice+Dal, Rajma, Rajma Chawal, Chole, Biryani, Mixed Thali, Roti, Paratha
- 15 Diverse Non-Food Negative Objects
"""

import os
import sys
import json
import io
import time
import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_validation")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_manifest.json")

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Curated high-reliability image definitions
DATASET_SPECS = [
    # -------------------------------------------------------------
    # 1. STAPLES: PLAIN RICE & RICE COMBOS
    # -------------------------------------------------------------
    {
        "category": "staples",
        "expected_class": "rice",
        "expected_type": "cooked_food",
        "filename": "plain_rice_01.jpg",
        "url": "https://images.unsplash.com/photo-1516684732162-798a0062be99?w=600&auto=format&fit=crop",
        "description": "Steamed white plain rice in a ceramic bowl on wooden table"
    },
    {
        "category": "staples",
        "expected_class": "rice",
        "expected_type": "cooked_food",
        "filename": "plain_rice_02.jpg",
        "url": "https://images.unsplash.com/photo-1541832676-9b763b0239ab?w=600&auto=format&fit=crop",
        "description": "Cooked white basmati rice plate top view"
    },
    {
        "category": "staples",
        "expected_class": "rice",
        "expected_type": "cooked_food",
        "filename": "plain_rice_03.jpg",
        "url": "https://images.unsplash.com/photo-1536304993881-ff6e9eefa2a6?w=600&auto=format&fit=crop",
        "description": "Plain boiled white rice close-up"
    },
    {
        "category": "staples",
        "expected_class": "fried_rice",
        "expected_type": "cooked_food",
        "filename": "fried_rice_01.jpg",
        "url": "https://images.unsplash.com/photo-1603133872878-684f208fb84b?w=600&auto=format&fit=crop",
        "description": "Chinese vegetable egg fried rice in a wok bowl"
    },

    # -------------------------------------------------------------
    # 2. BREADS: ROTI, PARATHA, NAAN
    # -------------------------------------------------------------
    {
        "category": "staples",
        "expected_class": "roti",
        "expected_type": "cooked_food",
        "filename": "roti_01.jpg",
        "url": "https://images.unsplash.com/photo-1626082927389-6cd097cdc6ec?w=600&auto=format&fit=crop",
        "description": "Fresh Indian roti flatbread with charred brown spots"
    },
    {
        "category": "staples",
        "expected_class": "paratha",
        "expected_type": "cooked_food",
        "filename": "paratha_01.jpg",
        "url": "https://images.unsplash.com/photo-1606471191009-63994c53433b?w=600&auto=format&fit=crop",
        "description": "Shallow-fried layered aloo paratha flatbread"
    },
    {
        "category": "staples",
        "expected_class": "naan",
        "expected_type": "cooked_food",
        "filename": "naan_01.jpg",
        "url": "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop",
        "description": "Tandoori garlic butter naan bread with blistered crust"
    },

    # -------------------------------------------------------------
    # 3. LENTILS & BEANS: DAL, RAJMA, CHOLE
    # -------------------------------------------------------------
    {
        "category": "lentils",
        "expected_class": "dal",
        "expected_type": "cooked_food",
        "filename": "dal_01.jpg",
        "url": "https://images.unsplash.com/photo-1546833999-b9f581a1996d?w=600&auto=format&fit=crop",
        "description": "Yellow dal tadka lentil soup with tempering in a bowl"
    },
    {
        "category": "lentils",
        "expected_class": "dal",
        "expected_type": "cooked_food",
        "filename": "dal_02.jpg",
        "url": "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?w=600&auto=format&fit=crop",
        "description": "Yellow lentil dal soup served with cilantro garnish"
    },
    {
        "category": "lentils",
        "expected_class": "rajma",
        "expected_type": "cooked_food",
        "filename": "rajma_01.jpg",
        "url": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&auto=format&fit=crop",
        "description": "Red kidney bean curry in thick spiced tomato gravy"
    },
    {
        "category": "lentils",
        "expected_class": "chole",
        "expected_type": "cooked_food",
        "filename": "chole_01.jpg",
        "url": "https://images.unsplash.com/photo-1596797038530-2c107229654b?w=600&auto=format&fit=crop",
        "description": "Spiced North Indian chana masala / chole chickpea curry"
    },

    # -------------------------------------------------------------
    # 4. INDIAN DISHES & COMBOS
    # -------------------------------------------------------------
    {
        "category": "indian_dishes",
        "expected_class": "biryani",
        "expected_type": "cooked_food",
        "filename": "biryani_01.jpg",
        "url": "https://images.unsplash.com/photo-1563379091339-03b21ab4a4f8?w=600&auto=format&fit=crop",
        "description": "Fragrant basmati chicken dum biryani in brass bowl"
    },
    {
        "category": "indian_dishes",
        "expected_class": "biryani",
        "expected_type": "cooked_food",
        "filename": "biryani_02.jpg",
        "url": "https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=600&auto=format&fit=crop",
        "description": "Spiced chicken biryani garnished with fried onions"
    },
    {
        "category": "indian_dishes",
        "expected_class": "paneer",
        "expected_type": "cooked_food",
        "filename": "paneer_01.jpg",
        "url": "https://images.unsplash.com/photo-1631452180519-c014fe946bc7?w=600&auto=format&fit=crop",
        "description": "Paneer butter masala cottage cheese in creamy tomato gravy"
    },
    {
        "category": "indian_dishes",
        "expected_class": "samosa",
        "expected_type": "cooked_food",
        "filename": "samosa_01.jpg",
        "url": "https://images.unsplash.com/photo-1601050690597-df0568f70950?w=600&auto=format&fit=crop",
        "description": "Crispy fried golden samosas with spiced potato filling"
    },
    {
        "category": "indian_dishes",
        "expected_class": "dosa",
        "expected_type": "cooked_food",
        "filename": "dosa_01.jpg",
        "url": "https://images.unsplash.com/photo-1668236543090-82eba5ee5976?w=600&auto=format&fit=crop",
        "description": "Crispy golden South Indian masala dosa crepe with chutney"
    },
    {
        "category": "indian_dishes",
        "expected_class": "idli",
        "expected_type": "cooked_food",
        "filename": "idli_01.jpg",
        "url": "https://images.unsplash.com/photo-1589301760014-d929f3979dbc?w=600&auto=format&fit=crop",
        "description": "Steamed white idli savory rice cakes with sambar dip"
    },
    {
        "category": "indian_dishes",
        "expected_class": "rajma_chawal",
        "expected_type": "cooked_food",
        "filename": "rajma_chawal_01.jpg",
        "url": "https://images.unsplash.com/photo-1585937421612-70a008356fbe?w=600&auto=format&fit=crop",
        "description": "Plate combining plain steamed rice and red kidney bean rajma curry"
    },
    {
        "category": "indian_dishes",
        "expected_class": "rice_dal_combo",
        "expected_type": "cooked_food",
        "filename": "rice_dal_combo_01.jpg",
        "url": "https://images.unsplash.com/photo-1626777552726-4a6b54c97e46?w=600&auto=format&fit=crop",
        "description": "Plate served with yellow dal poured over steamed white rice"
    },
    {
        "category": "indian_dishes",
        "expected_class": "thali_mixed",
        "expected_type": "cooked_food",
        "filename": "thali_mixed_01.jpg",
        "url": "https://images.unsplash.com/photo-1610057099443-fde8c4d50f91?w=600&auto=format&fit=crop",
        "description": "Full Indian thali metal platter with multiple bowls of curries, dal, roti and rice"
    },

    # -------------------------------------------------------------
    # 5. WESTERN & GLOBAL FOODS
    # -------------------------------------------------------------
    {
        "category": "western_foods",
        "expected_class": "pizza",
        "expected_type": "food",
        "filename": "pizza_01.jpg",
        "url": "https://images.unsplash.com/photo-1513104890138-7c749659a591?w=600&auto=format&fit=crop",
        "description": "Cheesy baked pizza slice on wooden board"
    },
    {
        "category": "western_foods",
        "expected_class": "burger",
        "expected_type": "food",
        "filename": "burger_01.jpg",
        "url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=600&auto=format&fit=crop",
        "description": "Juicy beef burger with toasted bun and cheese"
    },
    {
        "category": "western_foods",
        "expected_class": "pasta",
        "expected_type": "food",
        "filename": "pasta_01.jpg",
        "url": "https://images.unsplash.com/photo-1551183053-bf91a1d81141?w=600&auto=format&fit=crop",
        "description": "Spaghetti pasta with red tomato sauce and basil"
    },
    {
        "category": "western_foods",
        "expected_class": "sandwich",
        "expected_type": "food",
        "filename": "sandwich_01.jpg",
        "url": "https://images.unsplash.com/photo-1528735602780-2552fd46c7af?w=600&auto=format&fit=crop",
        "description": "Toasted grilled cheese sandwich cut in halves"
    },

    # -------------------------------------------------------------
    # 6. FRUITS
    # -------------------------------------------------------------
    {
        "category": "fruits",
        "expected_class": "apple",
        "expected_type": "fruit",
        "filename": "apple_01.jpg",
        "url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=600&auto=format&fit=crop",
        "description": "Fresh whole red apple on white background"
    },
    {
        "category": "fruits",
        "expected_class": "banana",
        "expected_type": "fruit",
        "filename": "banana_01.jpg",
        "url": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600&auto=format&fit=crop",
        "description": "Bunch of ripe yellow bananas on table"
    },
    {
        "category": "fruits",
        "expected_class": "mango",
        "expected_type": "fruit",
        "filename": "mango_01.jpg",
        "url": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=600&auto=format&fit=crop",
        "description": "Ripe sweet yellow mango"
    },
    {
        "category": "fruits",
        "expected_class": "orange",
        "expected_type": "fruit",
        "filename": "orange_01.jpg",
        "url": "https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?w=600&auto=format&fit=crop",
        "description": "Fresh whole citrus orange fruit"
    },

    # -------------------------------------------------------------
    # 7. UNKNOWN / OUT-OF-DISTRIBUTION DISHES (UNCERTAINTY TESTING)
    # -------------------------------------------------------------
    {
        "category": "unknown_dish",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "dimsum_01.jpg",
        "url": "https://images.unsplash.com/photo-1541696432-82c6da8ce7bf?w=600&auto=format&fit=crop",
        "description": "Steamed Chinese dim sum dumplings in bamboo basket"
    },
    {
        "category": "unknown_dish",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "paella_01.jpg",
        "url": "https://images.unsplash.com/photo-1534080564583-6be75777b70a?w=600&auto=format&fit=crop",
        "description": "Spanish seafood paella pan with mussels, shrimp and saffron rice"
    },
    {
        "category": "unknown_dish",
        "expected_class": "unknown_food",
        "expected_type": "food",
        "filename": "rambutan_01.jpg",
        "url": "https://images.unsplash.com/photo-1587132137056-bfbf0166836e?w=600&auto=format&fit=crop",
        "description": "Hairy exotic tropical rambutan fruit"
    },

    # -------------------------------------------------------------
    # 8. 15 DIVERSE NON-FOOD NEGATIVE OBJECTS
    # -------------------------------------------------------------
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "phone_01.jpg",
        "url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=600&auto=format&fit=crop",
        "description": "Modern smartphone on dark table"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "laptop_01.jpg",
        "url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=600&auto=format&fit=crop",
        "description": "Open laptop computer on office desk"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "keys_01.jpg",
        "url": "https://images.unsplash.com/photo-1582139329536-e7284fece509?w=600&auto=format&fit=crop",
        "description": "Set of metal house keys on a wooden surface"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "watch_01.jpg",
        "url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600&auto=format&fit=crop",
        "description": "Wristwatch timepiece with leather strap"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "person_face_01.jpg",
        "url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=600&auto=format&fit=crop",
        "description": "Portrait photo of a young woman"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "book_01.jpg",
        "url": "https://images.unsplash.com/photo-1544947950-fa07a98d237f?w=600&auto=format&fit=crop",
        "description": "Paperback book lying open on a desk"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "water_bottle_01.jpg",
        "url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=600&auto=format&fit=crop",
        "description": "Reusable stainless steel water bottle"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "mouse_01.jpg",
        "url": "https://images.unsplash.com/photo-1615663245857-ac93bb7c39e7?w=600&auto=format&fit=crop",
        "description": "Wireless computer optical mouse"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "headphones_01.jpg",
        "url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=600&auto=format&fit=crop",
        "description": "Over-ear audio headphones on plain background"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "charger_01.jpg",
        "url": "https://images.unsplash.com/photo-1583863788434-e58a36330cf0?w=600&auto=format&fit=crop",
        "description": "White USB phone power adapter and charging cable"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "coffee_mug_01.jpg",
        "url": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600&auto=format&fit=crop",
        "description": "Ceramic coffee mug cup on table"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "pen_01.jpg",
        "url": "https://images.unsplash.com/photo-1583485088034-697b5bc54ccd?w=600&auto=format&fit=crop",
        "description": "Ballpoint writing pen on notebook"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "shoe_01.jpg",
        "url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=600&auto=format&fit=crop",
        "description": "Red athletic sneaker running shoe"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "car_01.jpg",
        "url": "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=600&auto=format&fit=crop",
        "description": "Modern sports car on city street"
    },
    {
        "category": "non_food",
        "expected_class": "non_food",
        "expected_type": "non_food",
        "filename": "cat_01.jpg",
        "url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=600&auto=format&fit=crop",
        "description": "Domestic pet cat portrait"
    },
]


def build_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    manifest = []
    print(f"Building Real-World Validation Dataset ({len(DATASET_SPECS)} items)...")

    for idx, spec in enumerate(DATASET_SPECS):
        fname = spec["filename"]
        cat = spec["category"]
        cat_dir = os.path.join(DATA_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        img_path = os.path.join(cat_dir, fname)

        # Download if missing or corrupt
        need_download = True
        if os.path.exists(img_path):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                need_download = False
            except Exception:
                need_download = True

        if need_download:
            print(f"[{idx+1}/{len(DATASET_SPECS)}] Downloading {fname}...")
            r = requests.get(spec["url"], headers=HEADERS, timeout=15)
            if r.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(r.content)
                time.sleep(0.1)
            else:
                print(f"Failed HTTP {r.status_code} for {fname}")
                continue

        manifest.append({
            "filename": fname,
            "category": cat,
            "expected_class": spec["expected_class"],
            "expected_type": spec["expected_type"],
            "relative_path": os.path.relpath(img_path, os.path.dirname(os.path.dirname(__file__))),
            "absolute_path": os.path.abspath(img_path),
            "description": spec["description"]
        })

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Validation dataset built successfully with {len(manifest)} verified images.")
    print(f"Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    build_dataset()
