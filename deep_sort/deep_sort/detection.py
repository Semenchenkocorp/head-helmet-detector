# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    Класс работа с bounding box для задачи детекции
    box_coord - координаты коробки в формате (x,y,w,h)
    где x,y - координаты верхнего левого угла
    w,h - ширина и высота коробки
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        temp = self.tlwh.copy()
        temp[2:] += temp[:2]
        return temp
    
    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        temp = self.tlwh.copy()
        temp[:2] += temp[2:] / 2
        temp[2] /= temp[3]
        return temp