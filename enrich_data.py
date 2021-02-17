import os
from pycocotools.coco import COCO
from PIL import Image, ImageFilter, ImageEnhance


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


def write_xml(anno_path, type, filename, img_info, ann_info):
    anno_path = anno_path.replace('.xml', type+'.xml')
    filename = filename.replace('.jpg', type+'.jpg')
    new_head = headstr % (filename, int(img_info['width']), int(img_info['height']), 3)
    f = open(anno_path, "w")
    f.write(new_head)
    for a in ann_info:
        # class_name, xmin, ymin, xmax, ymax
        box = a['bbox']
        category_id = a['category_id']
        f.write(objstr% (CLASSES[category_id-1], int(box[0]), int(box[1]), int(box[0])+int(box[2])-1, int(box[1])+int(box[3])-1))
    f.write(tailstr)


def enhence(filename, img_info, ann_info):
    im_root = os.path.join('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/ori', filename)
    xml_root = os.path.join('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/VOC2007/Annotations', filename.replace('.jpg', '.xml'))
    im = Image.open(im_root)
    im = im.convert('RGB')
    enh_bri = ImageEnhance.Brightness(im)
    im_bri1 = enh_bri.enhance(1.5)
    im_bri2 = enh_bri.enhance(0.8)
    enh_sha = ImageEnhance.Sharpness(im)
    im_sharp1 = enh_sha.enhance(5.0)
    im_sharp2 = enh_sha.enhance(0.8)
    enh_con = ImageEnhance.Contrast(im)
    im_cont1 = enh_con.enhance(1.5)
    im_cont2 = enh_con.enhance(0.8)
    im_cont1.save(im_root.replace('.jpg', '_cont1.jpg'))
    write_xml(xml_root, '_cont1', filename, img_info, ann_info)
    im_cont2.save(im_root.replace('.jpg', '_cont2.jpg'))
    write_xml(xml_root, '_cont2', filename, img_info, ann_info)
    im_bri1.save(im_root.replace('.jpg', '_bri1.jpg'))
    write_xml(xml_root, '_bri1', filename, img_info, ann_info)
    im_bri2.save(im_root.replace('.jpg', '_bri2.jpg'))
    write_xml(xml_root, '_bri2', filename, img_info, ann_info)
    im_sharp1.save(im_root.replace('.jpg', '_sharp1.jpg'))
    write_xml(xml_root, '_sharp1', filename, img_info, ann_info)
    im_sharp2.save(im_root.replace('.jpg', '_sharp2.jpg'))
    write_xml(xml_root, '_sharp2', filename, img_info, ann_info)


def run(coco, enrich_id, keep):
    img_ids = coco.get_img_ids()
    num = 0
    txt_f = open('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train_enrich_1.txt', 'a')
    for img_id in img_ids:
        img_info = coco.load_imgs([img_id])[0]
        filename = img_info['file_name']
        ann_ids = coco.get_ann_ids(img_ids=[img_id])
        ann_info = coco.load_anns(ann_ids)
        category_ids = [a['category_id'] for a in ann_info]
        if enrich_id in category_ids and filename not in keep:
            keep.append(filename)
            num += 1
            print(enrich_id, num)
            # enhence(filename, img_info, ann_info)
            for tail in ['_cont1', '_cont2', '_bri1', '_bri2', '_sharp1', '_sharp2']:
                txt_f.write(filename.replace('.jpg', tail)+'\n')
    txt_f.close()
    print('{} images contain category_id {}'.format(num, enrich_id))
    return keep


def main():
    enrich_ids = [2, 13]
    keep = []
    coco = COCO('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/annotations/train.json')
    for enrich_id in enrich_ids:
        keep = run(coco, enrich_id, keep)


if __name__ == '__main__':
    main()
    # enhence('/home/user/workspace/lhz/vinbigdata/mmdetection/data/vinbigdata/002a34c58c5b758217ed1f584ccbcfe9.jpg')
