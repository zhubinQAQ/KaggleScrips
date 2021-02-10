sorted_csv_root = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/train_sorted.csv'

import csv
import numpy as np
from nms import NMS

headers = ['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height']

new_f = open('filtered_train.csv', 'w')
csv_writer = csv.writer(new_f)
csv_writer.writerow(headers)
nms = NMS(0.65)
filter_num = 0


def weighted_box(groups, csv_writer):
    global filter_num
    m_x1 = m_x2 = m_y1 = m_y2 = 0
    # print('ori image:', groups[0][0], groups[0][1])
    all = []
    for g in groups:
        x1, y1, x2, y2 = g[4:8]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        area = (x2-x1)*(y2-y1)
        all.append([x1, y1, x2, y2, area])
        # print(area, x1, y1, x2, y2)
        m_x1 += x1
        m_x2 += x2
        m_y1 += y1
        m_y2 += y2
    all = np.array(all)
    # print('before NMS')
    # print(all)
    before = len(all)
    all = all[nms.NMS1(all[:, :-1], all[:, -1])]
    after = len(all)
    filter_num += before - after
    base = groups[0]
    for i in range(after):
        base[4:8] = all[i][:4]
        csv_writer.writerow(base)
    # m_x1 /= len(groups)
    # m_x2 /= len(groups)
    # m_y1 /= len(groups)
    # m_y2 /= len(groups)
    # print('mean:')
    # print(m_x1, m_y1, m_x2, m_y2)


def filter_per_image(per_image, csv_writer):
    classes = {}
    for per_ins in per_image:
        class_id = int(per_ins[2])
        if class_id not in classes:
            classes[class_id] = [per_ins]
        else:
            classes[class_id].append(per_ins)

    for class_id, groups in classes.items():
        if len(groups) > 1:
            weighted_box(groups, csv_writer)
        else:
            csv_writer.writerow(groups[0])


with open(sorted_csv_root)as f:
    f_csv = csv.reader(f)
    num = 0
    id = ''
    per_image = []
    for row in f_csv:
        if num == 0 or row[2] == "14":
            num += 1
            continue
        if not len(per_image):
            id = row[0]
            per_image.append(row)
        if id == row[0]:
            per_image.append(row)
        else:
            filter_per_image(per_image, csv_writer)
            id = row[0]
            per_image = [row]
        num += 1
    print("Remove {} boxes".format(filter_num))

new_f.close()