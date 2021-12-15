import io
import os
from google.cloud import vision
import numpy as np
import cv2
import sys
import pandas as pd
import pprint
import xml.dom.minidom as md
import glob
import re
import concurrent.futures
from difflib import SequenceMatcher

brands: list = None


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1, bb2 = bb1['coords'], bb2['coords']

    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def filter_logos(logos, thresh=0.50):
    new_logos = []
    logos = sorted(logos, key=lambda x: -x['score'])

    for logo in logos:
        ious_with_logos = [0.0] + \
            [get_iou(logo, new_logo) for new_logo in new_logos]
        if max(ious_with_logos) > thresh:
            continue
        new_logos.append(logo)

    return new_logos


def convert_logos(logos, h, w, transform_coord_fn):
    def bounding_poly_to_cv2(bounding_poly):

        x0, y0, x1, y1 = 999999, 99999999, 0, 0

        for vertex in bounding_poly.vertices:
            x0 = min(x0, vertex.x)
            y0 = min(y0, vertex.y)
            x1 = max(x1, vertex.x)
            y1 = max(y1, vertex.y)

        x0, x1 = x0 / w, x1 / w
        y0, y1 = y0 / h, y1 / h

        y0, x0, y1, x1 = transform_coord_fn(np.array((y0, x0, y1, x1)))

        return y0, x0, y1, x1

    logos_list = []

    for logo in logos:
        logo_dict = {}
        logo_dict["brand"] = logo.description
        logo_dict['coords'] = bounding_poly_to_cv2(logo.bounding_poly)
        logo_dict['score'] = logo.score

        logos_list.append(logo_dict)

    return logos_list


def showimage(img):
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_logos(path, h, w, transform_coord_fn):
    global client, cache

    if path in cache:
        return cache[path]
    """Detects logos in the file."""

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response, err = None, None
    for i in range(10):
        try:
            response = client.logo_detection(image=image)
            break
        except Exception as e:
            err = e

    if response is None:
        raise err

    logos = response.logo_annotations
    logos = convert_logos(logos, h, w, transform_coord_fn)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    cache[path] = logos
    np.save(".cache_logos.npy", cache)
    return logos


def find_best_brand(logo_brand):
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    global brands
    results = []
    for br in brands:
        results.append((br, similar(br, logo_brand)))

    results = sorted(results, key = lambda x : -x[1])
    print(f"'{logo_brand}' was assigned '{results[0][0]}' with {results[0][1]:.2f} score")
    return results[0]

def process_image(image_path):
    _, brand, image_name = image_path.split("/")

    img = cv2.imread(image_path)
    try:
        height, width = img.shape[:2]
    except:
        return

    logos = []

    for fn_name, fn, transform_coord in zip(fns_names, fns, coord_transform_fn):
        new_img = fn(img)
        new_path = fn_name + '_' + image_name
        cv2.imwrite(new_path, new_img)

        try:
            logos += detect_logos(new_path, height, width, transform_coord)
        except Exception as e:
            print(f"Image '{image_path}' couldn't be processed: '{e}'")

        os.system(f"rm {new_path}")

    logos = [l for l in logos if l['score'] > 0.80]

    if len(logos) == 0:
        #os.remove(image_path)
        return

    logos = filter_logos(logos)

    root = md.parse('105113541.xml')
    objects = root.getElementsByTagName("object")
    for node in objects:
        parent = node.parentNode
        parent.removeChild(node)

    root.getElementsByTagName('filename')[0].firstChild.data = image_path
    root.getElementsByTagName('height')[0].firstChild.data = height
    root.getElementsByTagName('width')[0].firstChild.data = width

    for logo in logos:
        new_node = node.cloneNode(deep=True)

        new_node.getElementsByTagName("truncated")[0].firstChild.data = str(logo["score"] * 100)
        best_match_brand, score = find_best_brand(logo["brand"])
        if score < 0.6:
            continue

        new_node.getElementsByTagName("name")[0].firstChild.data = best_match_brand
        if logo["brand"] not in label_map:
            label_map[logo["brand"]] = 0
        label_map[logo["brand"]] += 1

        y0, x0, y1, x1 = (
            logo['coords'] * np.array([height, width, height, width])).astype("int")

        new_node.getElementsByTagName("xmin")[0].firstChild.data = x0
        new_node.getElementsByTagName("ymin")[0].firstChild.data = y0
        new_node.getElementsByTagName("xmax")[0].firstChild.data = x1
        new_node.getElementsByTagName("ymax")[0].firstChild.data = y1

        root.childNodes[0].appendChild(new_node)

    xml_str = root.toprettyxml() 
    xml_str = re.sub(r'\n[\s]*\n', '\n', xml_str)
    save_path_file = outfolder + '/' + brand.replace(" ", "") + "_" + image_name.split(".")[0] + ".xml"
    try:
        with open(save_path_file, "w") as f:
            f.write(xml_str) 
    except Exception as e:
        print(e)

    os.system(f"ln {image_path} {outfolder}/{brand.replace(' ', '')}_{image_name}")

    """
    if debug:
        try:
            img = tf.convert_to_tensor(
                np.expand_dims(img, 0) / 255, dtype="float32")
            boxes = tf.convert_to_tensor(np.expand_dims(
                np.array([l['coords'] for l in logos]), 0), dtype=tf.float32)
            colors = tf.convert_to_tensor(
                [[0.0, 0.0, 1.0] for x in enumerate(logos)], dtype=tf.float32)

            drawn = tf.image.draw_bounding_boxes(img, boxes, colors) if len(logos) else img
            drawn = (drawn.numpy() * 255).astype("uint8")[0]

            showimage(drawn)
        except Exception as e:
            print(e)

    """

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/andresarpi/.ssh/gcloud-keys.json'

OUTPATH = "popular_logos_binary"
INPATH = "popular_logos_plain"

client = vision.ImageAnnotatorClient()

try:
    cache = np.load('.cache_logos.npy', allow_pickle='TRUE').item()
except:
    print("Couldn't load cache.")
    cache = {}

if __name__ == "__main__":

    assert len(
        sys.argv) == 3, "This script needs to be called with 'python [filename].py [input dir] [output dir]' format"
    images_folder = sys.argv[1]
    outfolder = sys.argv[2]

    assert os.path.isdir(
        images_folder), f"'{images_folder}' is not a existing directory."

    os.makedirs(outfolder, exist_ok=True)

    brands = os.listdir(images_folder)
    present_images = glob.glob(f"{images_folder}/*/*", recursive = True)
    present_images = [pi for pi in present_images if pi.split(".")[-1] in ["jpg", "jpeg", "png"]]

    fns_names = ["identical", "reversed"]
    fns = [lambda x: x, lambda x: cv2.flip(x, 1)]
    coord_transform_fn = [lambda x: x, lambda x: (
        x[0], 1 - x[3], x[2], 1 - x[1])]

    for fn_name in fns_names:
        os.makedirs(fn_name, exist_ok=True)

    debug = False

    label_map = {}
    #for image_path in present_images:
     #   process_image(image_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers = 8) as executor:
        executor.map(process_image, present_images)


    with open("label_map.pbtxt", "w") as text_file:
        for i, (brand, count) in enumerate(label_map.items()):
            text_file.write("item {\n\tid: %d\n\tname: '%s'\n\tcount: %d\n}\n\n" % (i + 1, brand, count))


