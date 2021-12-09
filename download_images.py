import tqdm
import re
import os
import urllib.request
import urllib.error
import urllib.parse
import pandas as pd
import concurrent.futures
from serpapi import GoogleSearch
import datetime
import sys

params = {
  "api_key": "7f8127326245feb0a406ede628de5f9216511d1f1a8adf6bc80ed0bf4b97feee",
  "engine": "google",
  "q": "Coffee Starbucks",
  "tbm": "isch"
}

limit_searches = None
outpath = None

def convert_to_brand_path(brand):
    return re.sub(r"[^a-zA-z]", "-", brand)

def download_image(in_):
    global outpath
    image_link, brand = in_
    brand_path = os.path.join(outpath, convert_to_brand_path(brand))

    try:
        raw_img = urllib.request.urlopen(image_link).read()

        image_name = urllib.parse.urlsplit(image_link).path.split("/")[-1].split(".")[0]
        image_name = image_name[:min(12, len(image_name))] + ".jpg"

        f = open(os.path.join(brand_path, image_name), 'wb')
        f.write(raw_img)
        f.close()
    except Exception as e:
        pass#print(f"Could not download {image_link}: {str(e)}.")



def download_brand(brand):
    global limit_searches
    tqdm.tqdm.write(f"Starting {brand}")
    queries = [brand, f"{brand} logo", f"{brand} ad"]

    images_links = set()
    for query in queries:
        param = params.copy()
        param["q"] = query

        search = GoogleSearch(param)
        results = search.get_dict()

        for result in results['images_results']:
            try:
                images_links.add(result["original"])
            except:
                pass


    images_links = sorted(list(images_links))[:limit_searches]

    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(download_image, zip(images_links, [brand for x in range(len(images_links))]))

    downloaded_images = len(os.listdir(os.path.join(outpath, convert_to_brand_path(brand))))
    tqdm.tqdm.write(f"{datetime.datetime.now()} - Brand {brand} ended downloading {downloaded_images} out of {len(images_links)}")


def download_all_brands(brands):
    for brand in tqdm.tqdm(brands):
        download_brand(brand)

try:
    number_of_brands = int(sys.argv[1])
    limit_searches = int(sys.argv[2])
except:
    print("Call the script with 'python download_images.py [number of brands] [number of image per brand]")
    exit()

print(f"Will download at most {limit_searches} for {number_of_brands} brands")

outpath = "output"
brands_df = pd.read_csv('1000-brands.csv')
brands = brands_df["name"] #.str.replace(r'[^a-zA-Z]', '_', regex=True)
brands = brands[:]

os.makedirs(outpath, exist_ok=True)
for brand in brands:
    os.makedirs(os.path.join(outpath, convert_to_brand_path(brand)), exist_ok=True)


download_all_brands(brands)
