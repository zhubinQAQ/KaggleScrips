import json
from pycocotools.coco import COCO
infos = json.load(open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train.json'))

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

coco = COCO('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train.json')


f = open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train_enrich_1.txt')
lines = f.readlines()
count = {}
for l in lines:
    l = l.replace('\n', '')
    if '_' in l:
        l = l.split('_')[0]
    if l in count:
        count[l] += 1
    else:
        count[l] = 1

print(count)
nums = [0]*14
img_ids = coco.get_img_ids()
for img_id in img_ids:
    img_info = coco.load_imgs([img_id])[0]
    filename = img_info['file_name']
    c = count[filename.replace('.jpg', '')]
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    ann_info = coco.load_anns(ann_ids)
    for a in ann_info:
        nums[a['category_id']-1] += c

print(nums)
s = sum(nums)
for i in range(14):
    nums[i] /= s
print(nums)