import json

data_type = '_filter'
root = '/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/val{}.json'.format(data_type)
val_infos = json.load(open(root))
train_infos = json.load(open(root.replace('val{}.json'.format(data_type), 'train{}.json'.format(data_type))))
for t in ['val{}'.format(data_type), 'train{}'.format(data_type)]:
    f = open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/{}.txt'.format(t), 'w')
    infos = train_infos if t == 'train' else val_infos
    for image in infos['images']:
        f.write(image['file_name'][:-4] + '\n')
    f.close()

headstr = """\
<annotation>
    <folder>VOC</folder>
    <filename>%s</filename>
    <source>
        <database>My Database</database>
        <annotation>COCO</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''
CLASSES = (
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
        'Pulmonary fibrosis',
    )

import os
from pycocotools.coco import COCO


def write_xml(anno_path, head, ann_info):
    f = open(anno_path, "w")
    f.write(head)
    for a in ann_info:
        # class_name, xmin, ymin, xmax, ymax
        box = a['bbox']
        category_id = a['category_id']
        f.write(objstr% (CLASSES[category_id-1], int(box[0]), int(box[1]), int(box[0])+int(box[2])-1, int(box[1])+int(box[3])-1))
    f.write(tailstr)


for r in [root, root.replace('val{}.json'.format(data_type), 'train{}.json').format(data_type)]:
    coco = COCO(r)
    img_ids = coco.get_img_ids()
    num = 0
    for img_id in img_ids:
        num += 1
        img_info = coco.load_imgs([img_id])[0]
        filename = img_info['file_name']
        ann_ids = coco.get_ann_ids(img_ids=[img_id])
        ann_info = coco.load_anns(ann_ids)
        anno_path = os.path.join('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/VOC2007{}/Annotations'.format(data_type),filename.replace('.jpg', '.xml'))
        head = headstr % (filename, int(img_info['width']), int(img_info['height']), 3)
        write_xml(anno_path, head, ann_info)
        print(num)


