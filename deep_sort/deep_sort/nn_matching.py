# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Вычислите попарно квадрат расстояния между точками `а` и `в`.

    Параметры
    ----------
    a : array_like
        Матрица NxM из N выборок размерности M.
    b : array_like
        Матрица LxM из L выборок размерности M.

    Возвращается
    -------
    ndarray
        Возвращает матрицу размера len(a), len(b), такую, что элемент (i, j)
        содержит квадрат расстояния между `a[i]` и `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Вычислите попарное косинусное расстояние между точками `a` и `b`.

    Параметры
    ----------
    a : array_like
        Матрица NxM из N выборок размерности M.
    b : array_like
        Матрица LxM из L выборок размерности M.
    data_is_normalized : Необязательно[bool]
        Если значение равно True, предполагается, что строки в a и b являются векторами единичной длины.
        В противном случае a и b явно нормализованы к длине 1.

    Возвращается
    -------
    ndarray
        Возвращает матрицу размера len(a), len(b), такой, что элемент (i, j)
        содержит квадрат расстояния между `a[i]` и `b[j]`.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Вспомогательная функция для измерения расстояния до ближайшего соседа (евклидова).

    Параметры
    ----------
    x : ndarray
        Матрица из N векторов-строк (точек выборки).
    y : ndarray
        Матрица из M векторов-строк (точек запроса).

    Возвращается
    -------
    ndarray выделить
        Вектор длиной M, содержащий для каждой записи в "y"
наименьшее евклидово расстояние до выборки в `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Вспомогательная функция для определения расстояния до ближайшего соседа (косинус).

    Параметры
    ----------
    x : ndarray
        Матрица из N векторов-строк (точек выборки).
    y : ndarray
        Матрица из M векторов-строк (точек запроса).

    Возвращается
    -------
    ndarray выделить
        Вектор длиной M, содержащий для каждой записи в "y"
наименьшее косинусное расстояние до выборки в `x`.

    """
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    Показатель расстояния до ближайшего соседа, который для каждой цели возвращает значение
    ближайшего расстояния до любого образца, которое наблюдалось на данный момент.

    Параметры
    ----------
    метрика : структура
        Либо "евклидова", либо "косинусоидальная".
    matching_threshold: значение с плавающей точкой
        Порог соответствия. Образцы с большего расстояния считаются
 неверный матч.
    бюджет : опционально[инт]
        Если нет ни у кого, зафиксировать образцы в классе, чтобы на это самое количество. Удаляет
        древнейшие образцы, когда бюджет достиг.

    Атрибуты
    ----------
    примеры : Dict[int -> List[ndarray]]
        Словарь, который сопоставляет целевые идентификаторы со списком образцов
        , которые были просмотрены до сих пор.

    """

    def __init__(self, metric, matching_threshold, budget=None):


        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Обновите метрику расстояния новыми данными.

        Параметры
        ----------
        характеристики : ndarray
            Матрица NxM из N объектов размерности M.
        цели : ndarray
            Целочисленный массив связанных идентификаторов целей.
        active_targets : Список[int]
            Список целей, которые в данный момент присутствуют на месте происшествия.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Вычислите расстояние между объектами и целями.

        Параметры
        ----------
        features : ndarray
            Матрица NxM из N объектов размерности M.
        цели : Список[int]
            Список целей, с которыми необходимо сопоставить заданные `функции`.

        Возвращается
        -------
        ndarray выделить
            Возвращает матрицу затрат формы len(цели), len(объекты), где
            элемент (i, j) содержит ближайший квадрат расстояния между
            `целями[i]` и `объектами[j]`.

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix