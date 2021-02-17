#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 导入CSV安装包
import csv
import pickle as pickle
import numpy as np
import json
import argparse


image_infos = json.load(open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/test.json', 'r'))['images']
class_infos = csv.reader(open('/home/user/workspace/lhz/vinbigdata/mmdetection/download.csv'))
class_infos = {row[0]: row[1] for row in class_infos}


def run(inf, type_name, params):
    k1, k2 = params
    f = open('submission_{}_{}_{}.csv'.format(type_name, k1, k2), 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["image_id", "PredictionString"])
    add_num = replace_num = 0

    for image_result, image_info in zip(inf, image_infos):
        image_id = image_info['file_name'][:-4]
        line = ''
        max_score = 0

        for i in range(14):
            if len(image_result[i]):
                # if image_result[i][0][-1] > k:
                score = image_result[i][0][-1]
                max_score = max(max_score, score)
                box = image_result[i][0][:4]
                label = i
                line += "{} {} {} {} {} {} ".format(label, score, int(box[0]), int(box[1]), int(box[2]), int(box[3]))

        if type_name == 'hand' or type_name == 'none':
            if max_score < k2:
                replace_num += 1
                line = "{} {} {} {} {} {}".format(14, 1, 0, 0, 1, 1)
                csv_writer.writerow([image_id, line])
            elif max_score < k1:
                add_num += 1
                line += "{} {} {} {} {} {}".format(14, 1, 0, 0, 1, 1)
                csv_writer.writerow([image_id, line])
            else:
                csv_writer.writerow([image_id, line])
        elif type_name == 'classifier':
            class_score = float(class_infos[image_id])
            if class_score >= k1:
                line = "{} {} {} {} {} {}".format(14, 1, 0, 0, 1, 1)
                replace_num += 1
                csv_writer.writerow([image_id, line])
            elif class_score >= k2:
                add_num += 1
                line += "{} {} {} {} {} {}".format(14, class_score, 0, 0, 1, 1)
                csv_writer.writerow([image_id, line])
            else:
                csv_writer.writerow([image_id, line])

    print('add: {}, replace: {}'.format(add_num, replace_num))

    f.close()


def main():
    parser = argparse.ArgumentParser(description='make csv')
    parser.add_argument('bbox_file', help='train config file path')
    args = parser.parse_args()
    params = {
        'none': (0.0, 0.0),
        'hand': (0.91, 0.0),
        'classifier': (0.97, 0.0),
    }
    fr = open(args.bbox_file, 'rb')  # open的参数是pkl文件的路径
    inf = pickle.load(fr)
    for t in ['none', 'hand', 'classifier']:
        run(inf, t, params[t])


if __name__ == '__main__':
    main()