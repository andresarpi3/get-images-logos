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
from timeout import timeout
import numpy as np
import threading
import cv2
from google.cloud import vision
from tag_helper import *

from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv('SERPAPI_KEY')


brand_counts = {}
brand_counts_lock = threading.Lock()

serpapi_cache = None
try:
    serpapi_cache = np.load(".serpapi_cache.npy", allow_pickle=True).item()
except Exception as e:
    serpapi_cache = {}
serpapi_cache_lock = threading.Lock()


params = {
  "api_key": SERPAPI_KEY,
  "engine": "google",
  "q": "Coffee Starbucks",
  "tbm": "isch"
}

fns_names = ["identical", "reversed"]
fns = [lambda x: x, lambda x: cv2.flip(x, 1)]
coord_transform_fn = [lambda x: x, lambda x: (
    x[0], 1 - x[3], x[2], 1 - x[1])]


limit_searches = None
outpath = None

def convert_to_brand_path(brand):
    return re.sub(r"[^a-zA-z]", "-", brand)

def download_image(in_):
    global outpath
    image_link, brand, idx = in_
    brand_path = os.path.join(outpath, convert_to_brand_path(brand))

    try:
        image_name = urllib.parse.urlsplit(image_link).path.split("/")[-1].split(".")[0]
        idx = hash(image_name)
        image_name = f"{brand}_{idx:04d}.jpg"

        if os.path.isfile(os.path.join(brand_path, image_name)):
            return

        raw_img = urllib.request.urlopen(image_link).read()

        f = open(os.path.join(brand_path, image_name), 'wb')
        f.write(raw_img)
        f.close()
    except Exception as e:
        pass #print(f"Could not download {image_link}: {str(e)}.")

def find_how_many_images_have_correct_logos(brand, brands):
    global outpath


    present_logos_for_brand = 0
    brand_path = os.path.join(outpath, convert_to_brand_path(brand))
    for present_image in os.listdir(brand_path):
        present_image_path = os.path.join(brand_path, present_image)
        logos_found = find_logos(present_image_path, 0.5, brands)
        logos_found_for_brand = [l for l in logos_found if l['brand'] == brand]
        present_logos_for_brand += len(logos_found_for_brand)

    return present_logos_for_brand

def copy_patches(brand, brands):
    global patches_outpath

    brand_path = os.path.join(outpath, convert_to_brand_path(brand))
    for present_image in os.listdir(brand_path):
        present_image_path = os.path.join(brand_path, present_image)
        logos_found = find_logos(present_image_path, 0.5, brands)
        if len(logos_found) == 0:
            continue
        img = cv2.imread(present_image_path)
        h, w = img.shape[:2]
        for patch_idx, logo in enumerate(logos_found):
            cs = logo["coords"]
            y0, x0, y1, x1 = np.array([cs[0] * h, cs[1] * w, cs[2] * h, cs[3] * w]).astype("int")
            patch = img[y0:y1, x0:x1]
            patch_path = os.path.join(patches_outpath, convert_to_brand_path(brand), f"{present_image.split('.')[0]}_{patch_idx}.jpg")
            cv2.imwrite(patch_path, patch)

def generate_xmls(brand, brands):
    global xmls_outpath

    brand_path = os.path.join(outpath, convert_to_brand_path(brand))
    for present_image in os.listdir(brand_path):
        present_image_path = os.path.join(brand_path, present_image)
        logos_found = find_logos(present_image_path, 0.5, brands)
        if len(logos_found) == 0:
            continue
        
        xml_str = create_xml_with_logos(logos_found, present_image_path)
        xml_path_file = os.path.join(xmls_outpath, convert_to_brand_path(brand), f"{present_image.split('.')[0]}.xml")
        new_image_path = os.path.join(xmls_outpath, convert_to_brand_path(brand), f"{present_image.split('.')[0]}.jpg")
        try:
            with open(xml_path_file, "w") as f:
                f.write(xml_str) 
        except Exception as e:
            print(e)

        os.system(f"ln {present_image_path} {new_image_path}")

def download_brand(brand):

    global limit_searches, serpapi_cache, outpath, brands

    tqdm.tqdm.write(f"Starting {brand}")
    queries = [brand, f"{brand} logo", f"{brand} ad"]

    # Get Images links
    images_links = set()
    for query in queries:
        param = params.copy()
        param["q"] = query

        if query in serpapi_cache:
            results = serpapi_cache[query]
        else:
            search = GoogleSearch(param)
            results = search.get_dict()

            serpapi_cache_lock.acquire()
            try:
                serpapi_cache[query] = results
                np.save(".serpapi_cache.npy", serpapi_cache)
            except:
                pass
            finally:
                serpapi_cache_lock.release()


        for result in results['images_results']:
            try:
                images_links.add(result["original"])
            except:
                pass

    images_links = list(images_links)
    np.random.shuffle(images_links)

    ### Download Images links
    @timeout(limit_searches * 2)
    def download_images(limit_searches, images_links, brand, idxs):
        """
        for arg in zip(images_links, [brand for x in range(len(images_links))], idxs):
            download_image(arg)

        return
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers = limit_searches) as executor:
            executor.map(download_image, zip(images_links, [brand for x in range(len(images_links))], idxs))


    #See how many images with logos we already have
    runs = 0
    while find_how_many_images_have_correct_logos(brand, brands) < limit_searches and runs < 10:
        runs += 1
        #tqdm.tqdm.write(f"{datetime.datetime.now()} - {brand} - Starting run {runs} with \
        #{find_how_many_images_have_correct_logos(brand, brands)} valid images \
        #({len(os.listdir(os.path.join(outpath, convert_to_brand_path(brand))))} total images).")

        try:
            start, end = (runs - 1) * limit_searches, runs * limit_searches
            if end > len(images_links):
                break
            download_images(limit_searches, images_links[start:end], brand, list(range(start, end)))
        except Exception as e:
            print(f"{brand} ran out of time...")

    downloaded_images = find_how_many_images_have_correct_logos(brand, brands)
    copy_patches(brand, brands)
    generate_xmls(brand, brands)
    tqdm.tqdm.write(f"{datetime.datetime.now()} - {brand} - Leaves with {downloaded_images} valid images \
        ({len(os.listdir(os.path.join(outpath, convert_to_brand_path(brand))))} total images).")

def download_all_brands(brands):
    global limit_searches
    
    def dms(brands):
        for brand in tqdm.tqdm(brands):
            download_brand(brand)

    try:
        dms(brands)
    except Exception as e:
        if type(e) == TimeoutError:
            print("Run took too long, starting another one...")
        else:
            raise e


if __name__ == "__main__":


    try:
        number_of_brands = int(sys.argv[1])
        limit_searches = int(sys.argv[2])
        patches_outpath = sys.argv[3]
        xmls_outpath = sys.argv[4]
    except:
        print("Call the script with 'python download_images.py [number of brands] [number of image per brand] \
              [patches outpath]")
        exit()


    outpath = "output"
    brands_df = pd.read_csv('1000-brands.csv')
    brands = brands_df["name"] 
    brands = brands[:number_of_brands]
    brands = [convert_to_brand_path(b) for b in brands]

    print(f"Will download at most {limit_searches} for {len(brands)} brands")

    os.makedirs(outpath, exist_ok=True)
    for brand in brands:
        os.makedirs(os.path.join(outpath, convert_to_brand_path(brand)), exist_ok=True)

    fns_names_directories = [os.path.join("to_delete", fnn) for fnn in fns_names]
    for fn_name in fns_names_directories:
        os.makedirs(fn_name, exist_ok=True)

    #np.random.shuffle(brands)
    os.makedirs(patches_outpath, exist_ok=True)
    _ = [os.makedirs(os.path.join(patches_outpath, b), exist_ok=True) for b in brands]

    os.makedirs(xmls_outpath, exist_ok=True)
    _ = [os.makedirs(os.path.join(xmls_outpath, b), exist_ok=True) for b in brands]


    download_all_brands(brands)


