import os
import csv
import json
import cv2
import numpy as np

from tqdm import tqdm

G_FILTERED = False
LABELS = ['No finding',
          'Aortic enlargement',
          'Atelectasis',
          'Calcification',
          'Cardiomegaly',
          'Consolidation',
          'ILD',
          'Infiltration',
          'Lung Opacity',
          'Nodule/Mass',
          'Other lesion',
          'Pleural effusion',
          'Pleural thickening',
          'Pneumothorax',
          'Pulmonary fibrosis']
PATH_ANNOS = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/'
PATH_ORI_CSV = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/train.csv'
PATH_FILTERED_CSV = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/train_filter.csv'
PATH_TEST_IM = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/test'

filt = '_filter' if G_FILTERED else ''


def make_label_map():
    map1 = map2 = {}
    for i in range(len(LABELS)):
        map1[LABELS[i]] = i
        map2[i] = LABELS[i]
    return map1, map2


label_map, id_label_map = make_label_map()


def save_json(images, categories, annotations, save_json_path):
    data_coco = {'images': images, 'categories': categories, 'annotations': annotations}
    print('[SAVE] {}'.format(save_json_path))
    json.dump(data_coco, open(save_json_path, 'w'))


def origin_csv2json():
    print('==================[origin_csv2json]=====================')
    print('| convert origin train val csv to coco json             |\n')
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}

        categories.append(categorie)

    img_id = 0
    img_anno = {}
    with open(PATH_ORI_CSV)as f:
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
    save_json_path = os.path.join(PATH_ANNOS, 'train_val{}.json'.format(filt))
    save_json(images, categories, annotations, save_json_path)


def get_mean_std(root):
    print('==================[get_mean_std]=====================')
    print('| get mean and std of images                         |\n')
    root_list = os.listdir(root)
    print("{} -> {} imgs".format(root, len(root_list)))
    means = []
    stdevs = []

    img_list = []
    img_h, img_w = 32, 48  # 根据自己数据集适当调整，影响不大
    path = '/home/zhubin/data/512/vinbigdata/train'
    for item in tqdm(root_list, desc='get mean & std ..'):
        img = cv2.imread(os.path.join(path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))


def divide_train_val(ratio, json_file):
    from pycocotools.coco import COCO
    coco = COCO(json_file)
    ids = sorted(list(coco.imgs.keys()))

    split = int(len(ids) * ratio)
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
    root = os.path.join(PATH_ANNOS, "info{}.json".format(filt))
    json.dump(data_info, open(root, 'w'))


def convert_train_val(data_type, info):
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}

        categories.append(categorie)

    img_id = 0
    img_anno = {}
    csv_path = PATH_FILTERED_CSV if G_FILTERED else PATH_ORI_CSV

    with open(csv_path)as f:
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
    save_json_path = os.path.join(PATH_ANNOS, '{}{}.json'.format(data_type, filt))
    save_json(images, categories, annotations, save_json_path)


def make_test_json():
    print('==================[make_test_json]=====================')
    print('| in order to get bbox.pkl of test images, convert test|\n'
          '| images to coco json format.                          |\n')
    images = []
    categories = []
    annotations = []

    for i in range(1, len(label_map)):
        categorie = {'supercategory': id_label_map[i], 'id': i, 'name': id_label_map[i]}
        categories.append(categorie)

    img_id = 0
    img_anno = {}
    dirs = os.listdir(PATH_TEST_IM)
    num = 0
    for d in dirs:
        print(num)
        filename = d
        img = cv2.imread(os.path.join(PATH_TEST_IM, filename))
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
    save_json_path = os.path.join(PATH_ANNOS, 'test.json')
    save_json(images, categories, annotations, save_json_path)


def main():
    origin_csv2json()
    divide_train_val(0.8, os.path.join(PATH_ANNOS, 'train_val.json'))
    infos = json.load(open(os.path.join(PATH_ANNOS, 'info.json')))
    for data_type in ["train", "val"]:
        convert_train_val(data_type, infos[data_type])
    # make_test_json()


if __name__ == '__main__':
    main()