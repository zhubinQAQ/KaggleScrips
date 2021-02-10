#!/usr/bin/python3
# -*- coding: utf-8 -*-

# 导入CSV安装包
import csv
import pickle as pickle
import numpy as np
import json

fr = open('/home/user/workspace/lhz/vinbigdata/mmdetection/bbox.pkl','rb')    #open的参数是pkl文件的路径
inf = pickle.load(fr)
image_infos = json.load(open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/test.json', 'r'))['images']

print(len(inf))
k = 0.95
f = open('12_6_{}.csv'.format(k), 'w')
csv_writer = csv.writer(f)
csv_writer.writerow(["image_id", "PredictionString"])

for image_result, image_info in zip(inf, image_infos):
    image_id = image_info['file_name'][:-4]
    line = ''
    max_score = 0
    label = 14
    box = [0, 0, 1, 1]
    for i in range(14):
        if len(image_result[i]):
            if image_result[i][0][-1] > max_score:
                max_score = image_result[i][0][-1]
                box = image_result[i][0][:4]
                label = i
    if max_score < k:
        label = 14
        max_score = 1
        box = [0, 0, 1, 1]
    line = "{} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(label, 1, box[0], box[1], box[2], box[3])
    csv_writer.writerow([image_id, line])

    # for i in range(14):
    #     if len(image_result[i]):
    #         if image_result[i][0][-1] > k:
    #             max_score = image_result[i][0][-1]
    #             box = image_result[i][0][:4]
    #             label = i
    #             line = "{} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(label, max_score, box[0], box[1], box[2], box[3])
    #             csv_writer.writerow([image_id, line])
    #
    # if not len(line):
    #     line = "{} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}".format(label, 1, box[0], box[1], box[2], box[3])
    #     csv_writer.writerow([image_id, line])


f.close()
# 1. 创建文件对象
# f = open('10_1.csv','w')
#
# # 2. 基于文件对象构建 csv写入对象
# csv_writer = csv.writer(f)
#
# # 3. 构建列表头
# csv_writer.writerow(["image_id","PredictionString"])
#
# # 4. 写入csv文件内容
# csv_writer.writerow(["l",'18','男'])
# csv_writer.writerow(["c",'20','男'])
# csv_writer.writerow(["w",'22','女'])
#
# # 5. 关闭文件
# f.close()