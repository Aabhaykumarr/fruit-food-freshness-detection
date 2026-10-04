"""
Fetches authoritative page images from Wikipedia API for benchmark items.
"""

import os
import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_validation")

ITEMS = [
    {"title": "Chapati", "dest": "staples/roti_01.jpg"},
    {"title": "Paratha", "dest": "staples/paratha_01.jpg"},
    {"title": "Naan", "dest": "staples/naan_01.jpg"},
    {"title": "Cooked rice", "dest": "staples/plain_rice_02.jpg"},
    {"title": "Dal", "dest": "lentils/dal_01.jpg"},
    {"title": "Rajma", "dest": "lentils/rajma_01.jpg"},
    {"title": "Chana masala", "dest": "lentils/chole_01.jpg"},
    {"title": "Biryani", "dest": "indian_dishes/biryani_01.jpg"},
    {"title": "Thali", "dest": "indian_dishes/thali_mixed_01.jpg"},
]

HEADERS = {"User-Agent": "FruitFreshnessBot/1.0 (contact: test@example.com)"}


def fetch_wiki_images():
    for item in ITEMS:
        title = item["title"]
        dest = os.path.join(DATA_DIR, item["dest"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Fetching Wikipedia image for '{title}'...")

        api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=pageimages&format=json&pithumbsize=600"
        try:
            r = requests.get(api_url, headers=HEADERS, timeout=10)
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            img_url = None
            for pid, pdata in pages.items():
                if "thumbnail" in pdata:
                    img_url = pdata["thumbnail"]["source"]
                    break

            if img_url:
                print(f"  Downloading: {img_url}")
                img_r = requests.get(img_url, headers=HEADERS, timeout=15)
                if img_r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(img_r.content)
                    with Image.open(dest) as img:
                        img.verify()
                    print(f"  [SUCCESS] Verified and saved to {dest}")
                else:
                    print(f"  [ERROR] HTTP {img_r.status_code} downloading image")
            else:
                print(f"  [ERROR] No thumbnail found for page '{title}'")
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    fetch_wiki_images()
