"""
Searches Wikipedia for Dal tadka and Thali images.
"""

import os
import requests
from PIL import Image

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "real_world_validation")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) FruitDetectionResearch/1.0"}


def search_and_download(search_term, dest_rel):
    dest = os.path.join(DATA_DIR, dest_rel)
    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_term}&format=json"
    r = requests.get(search_url, headers=HEADERS).json()
    results = r.get("query", {}).get("search", [])
    for res in results[:3]:
        page_title = res["title"]
        api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={page_title}&prop=pageimages&format=json&pithumbsize=600"
        page_r = requests.get(api_url, headers=HEADERS).json()
        pages = page_r.get("query", {}).get("pages", {})
        for pid, pdata in pages.items():
            if "thumbnail" in pdata:
                img_url = pdata["thumbnail"]["source"]
                print(f"[{search_term}] Found on '{page_title}': {img_url}")
                img_r = requests.get(img_url, headers=HEADERS)
                if img_r.status_code == 200:
                    with open(dest, "wb") as f:
                        f.write(img_r.content)
                    with Image.open(dest) as img:
                        img.verify()
                    print(f"  [SAVED] {dest}")
                    return


if __name__ == "__main__":
    search_and_download("Tarka dal yellow", "lentils/dal_01.jpg")
    search_and_download("South Indian Thali meal", "indian_dishes/thali_mixed_01.jpg")
