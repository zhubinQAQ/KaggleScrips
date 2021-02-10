import numpy as np


class NMS:
    def __init__(self, thresh):
        self.thresh = thresh

    def NMS1(self, dets, score):
        '''
        按照score进行降序排序，然后每次拿到第一个框，也就是score最大的框，
        然后计算该框与其他框的IOU，最后留下iou<=thresh的框留作下次循环，
        这里唯一值得强调的是最后这个索引为什么要+1。这是因为我们要得到的inds
        是排除了当前用来比较的score最大的框，所以在其原始索引基础上+1,
        从代码中看就是由于order[1:]这样写导致的。
        '''
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)                      # 求每个矩形框的面积
        order = score.argsort()[::-1]                              # 按照从高到低的顺序排序
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])                 # 取当前框和后面框的最大最小值
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)    # 求取IoU
            inds = np.where(ovr <= self.thresh)[0]                 # 若IoU值大于阈值则说明重合度高，删去
            order = order[inds + 1]
        return keep

    def NMS2(self, bbox, score):
        '''
        同样先将框按score排好序，这次循环条件为遍历所有框，在某次循环，拿到一个框，
        将其与所有已经保留的框进行比较，如果iou大于阈值，说明它应该被删去，直接进行
        下一次循环，如果小于将其加入进要保留的框中。当遍历完所有框时结束，拿到保留的框。
        '''
        order = np.argsort(score)[::-1]
        bbox = bbox[order]
        bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)
        keep = np.zeros(bbox.shape[0], dtype=bool)
        for i, b in enumerate(bbox):
            tl = np.maximum(b[:2], bbox[keep, :2])
            br = np.minimum(b[2:], bbox[keep, 2:])
            area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)
            iou = area / (bbox_area[i] + bbox_area[keep] - area)
            if (iou >= self.thresh).any():
                continue
            keep[i] = True
        keep = np.where(keep)[0]
        return keep.astype(np.int32)

    # 非排序 NMS 的实现抛弃来排序，结果和排序的不一致但是对精度影响很小，速度会快一些
    def FastNMS(self, bbox, score):
        '''
        依次遍历每个框，计算这个框与其他框的iou,找到iou大于一定阈值的其他框，
        因为这个时候不能保证它一定是score最高的框，所以要进行判断，如果它的
        score小于其他框，那就把它去掉，因为它肯定不是要保留的框。如果它的score
        大于其他框，那应该保留它，同时可以去掉所有其他框了。
        '''
        area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)  # 求面积。np.prod是求积
        keep = np.ones(len(bbox), dtype=bool)              # 默认全保留
        for i, b in enumerate(bbox):
            if not keep[i]:
                continue
            tl = np.maximum(b[:2], bbox[i + 1:, :2])      # 依旧是常规的求i与i之后的框的iou得分
            br = np.minimum(b[2:], bbox[i + 1:, 2:])
            inter = np.prod(br - tl, axis=1) * (br >= tl).all(axis=1)
            iou = inter / (area[i + 1:] + area[i] - inter)
            r = [k for k in np.where(iou > self.thresh)[0] + i + 1 if keep[k]]
            if (score[i] > score[r]).all():
                keep[r] = False
            else:
                keep[i] = False
        return np.where(keep)[0].astype(np.int32)