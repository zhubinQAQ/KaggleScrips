cvs_path = "/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/"
import os
import csv
import json
import cv2
import numpy as np
from tqdm import tqdm


headers = ['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height']
label_map = {
    'No finding': 0,
    'Pulmonary fibrosis': 14,
    'Pneumothorax': 13,
    'Pleural thickening': 12,
    'Pleural effusion': 11,
    'Other lesion': 10,
    'Nodule/Mass': 9,
    'Lung Opacity': 8,
    'Infiltration': 7,
    'ILD': 6,
    'Consolidation': 5,
    'Cardiomegaly': 4,
    'Calcification': 3,
    'Atelectasis': 2,
    'Aortic enlargement': 1,
}

id_label_map = {v: k for k, v in label_map.items()}


def save_json(images, categories, annotations, save_json_path):
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    json.dump(data_coco, open(save_json_path, 'w'))


def convert(data_type='train'):
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}

        categories.append(categorie)

    img_id = 0
    img_anno = {}
    with open(cvs_path + '{}.csv'.format(data_type))as f:
        f_csv = csv.reader(f)
        num = 0
        for row in f_csv:
            if num == 0:
                num += 1
                continue
            if row[1] == 'No finding':
                continue
            filename = row[0] + '.jpg'
            width = row[8]
            height = row[9]
            if filename not in img_anno:
                img_id += 1
                img_anno[filename] = img_id
                # images
                image = {'height': height, 'width': width, 'id': img_id, 'file_name': filename}
                images.append(image)

            cur_img_id = img_anno[filename]
            # annotations
            label_id = label_map[row[1]]
            # if row[1] == 'No finding':
            #     continue
            #     x1 = y1 = 0.0
            #     x2 = y2 = 0.0
            # else:
            x1 = float(row[4])
            y1 = float(row[5])
            x2 = float(row[6])
            y2 = float(row[7])
            w = (x2 - x1 + 1)
            h = (y2 - y1 + 1)
            bbox = [x1, y1, w, h]
            area = w * h
            annotation = {'segmentation': [], 'iscrowd': 0, 'area': area, 'image_id': cur_img_id,
                          'bbox': bbox, 'difficult': 0,
                          'category_id': label_id, 'id': num}
            annotations.append(annotation)
            num += 1
    print("lines is {}".format(num))
    save_json_path = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/{}_val.json'.format(data_type)
    save_json(images, categories, annotations, save_json_path)


def get_images_info(root):
    root_list = os.listdir(root)
    print("{} -> {} imgs".format(root, len(root_list)))
    # means = []
    # stdevs = []
    #
    # img_list = []
    # img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
    # path = '/home/zhubin/data/512/vinbigdata/train'
    # for item in tqdm(root_list, desc='get mean & std ..'):
    #     img = cv2.imread(os.path.join(path, item))
    #     img = cv2.resize(img, (img_w, img_h))
    #     img = img[:, :, :, np.newaxis]
    #     img_list.append(img)
    #
    # imgs = np.concatenate(img_list, axis=3)
    # imgs = imgs.astype(np.float32) / 255.
    #
    # for i in range(3):
    #     pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    #     means.append(np.mean(pixels))
    #     stdevs.append(np.std(pixels))
    # # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    # means.reverse()
    # stdevs.reverse()
    #
    # print("normMean = {}".format(means))
    # print("normStd = {}".format(stdevs))


def divide_train_val(ratio, json_file):
    from pycocotools.coco import COCO
    coco = COCO(json_file)
    ids = sorted(list(coco.imgs.keys()))

    split = int(len(ids)*ratio)
    train_file = []
    val_file = []
    for id in tqdm(ids[:split], desc='iterate train ..'):
        path = coco.loadImgs(id)[0]['file_name']
        train_file.append(path)

    for id in tqdm(ids[split:], desc='iterate test ..'):
        path = coco.loadImgs(id)[0]['file_name']
        val_file.append(path)

    print(split, len(ids), len(train_file), len(val_file))
    data_info = {'train': train_file, 'val': val_file}
    root = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/'
    json.dump(data_info, open(root+"info.json", 'w'))


def convert_train_val(data_type, info):
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}

        categories.append(categorie)

    img_id = 0
    img_anno = {}
    cvs_path = "/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/train.csv"

    with open(cvs_path)as f:
        f_csv = csv.reader(f)
        num = 0
        for row in f_csv:
            if row[0] + '.jpg' not in info:
                continue
            if num == 0:
                num += 1
                continue
            if row[1] == 'No finding':
                continue
            filename = row[0] + '.jpg'
            width = int(row[8])
            height = int(row[9])
            if filename not in img_anno:
                img_id += 1
                img_anno[filename] = img_id
                # images
                image = {'height': height, 'width': width, 'id': img_id, 'file_name': filename}
                images.append(image)

            cur_img_id = img_anno[filename]
            # annotations
            label_id = label_map[row[1]]
            # if row[1] == 'No finding':
            #     continue
            #     x1 = y1 = 0.0
            #     x2 = y2 = 0.0
            # else:
            x1 = float(row[4])
            y1 = float(row[5])
            x2 = float(row[6])
            y2 = float(row[7])
            w = (x2 - x1 + 1)
            h = (y2 - y1 + 1)
            bbox = [x1, y1, w, h]
            area = w * h
            annotation = {'segmentation': [], 'iscrowd': 0, 'area': area, 'image_id': cur_img_id,
                          'bbox': bbox, 'difficult': 0,
                          'category_id': label_id, 'id': num}
            annotations.append(annotation)
            num += 1
    print("lines is {}".format(num))
    save_json_path = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/{}.json'.format(data_type)
    save_json(images, categories, annotations, save_json_path)


def get_test_json():
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}
        categories.append(categorie)

    img_id = 0
    img_anno = {}
    test_root = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/test'
    dirs = os.listdir(test_root)
    num = 0
    for d in dirs:
        print(num)
        filename = d
        img = cv2.imread(os.path.join(test_root, filename))
        height, width, c = img.shape
        if filename not in img_anno:
            img_id += 1
            img_anno[filename] = img_id
            # images
            image = {'height': height, 'width': width, 'id': img_id, 'file_name': filename}
            images.append(image)

        cur_img_id = img_anno[filename]
        annotation = {'segmentation': [], 'iscrowd': 0, 'area': 0, 'image_id': cur_img_id,
                      'bbox': [0, 0, 1, 1], 'difficult': 0,
                      'category_id': 0, 'id': num}
        annotations.append(annotation)
        num += 1
    print("lines is {}".format(num))
    save_json_path = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/test.json'
    save_json(images, categories, annotations, save_json_path)

# get_images_info('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/ori')
#
# convert()
# divide_train_val(0.8, '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train_val.json')
#
#
# infos = json.load(open("/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/info.json"))
# for data_type in ["train", "val"]:
#     convert_train_val(data_type, infos[data_type])


# get_test_json()
# filter_box()