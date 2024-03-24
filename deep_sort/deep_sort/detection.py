# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    Класс работа с bounding box для задачи детекции
    box_coord - координаты коробки в формате (x,y,w,h)
    где x,y - координаты верхнего левого угла
    w,h - ширина и высота коробки
    """

    def __init__(self, box_coord, confidence, feature):
        self.box_coord = np.asarray(box_coord, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """
        у нас подается бокс форматом (x,y,w,h) где x,y - координаты левого верхнего угла
        соответственно в этой функции мы получаем правый нижний угол
        """
        temp = self.box_coord.copy()
        temp[2:] += temp[:2]
        return temp

    def to_xyah(self):
        """
        бокс формата (x,y,w,h) преоборазовывается в формат
        (x_center, y_center, a, h)
        Где a это соотнощение ширины и длины
        """
        temp = self.box_coord.copy()
        temp[:2] += temp[2:] / 2
        temp[2] /= temp[3]
        return temp