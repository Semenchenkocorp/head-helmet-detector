# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Тип перечисления для единственного целевого состояния трека. Вновь созданные треки
классифицируются как "предварительные" до тех пор, пока не будет собрано достаточно доказательств. Затем
состояние трека изменяется на `подтвержденный`. Треки, которые больше не существуют
    классифицируются как `удаленные", чтобы пометить их для удаления из набора активных
    треков.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Интеллектуальная система "(x, y, a, h)" и соавторами
скоростями, где "(x, y)" - это ограничивающая линейка, "а" -
соединение строк, буква "н" - высота.

    Параметры
    ----------
    среднее значение : ndarray
        Вектор среднего значения распределения начального состояния.
    ковариация : ndarray
        Ковариационная матрица распределения начальных состояний.
    track_id : int
- уникальный идентификатор трека.
    n_init : int
- количество последовательных обнаружений до подтверждения трека. То
        состояние трека устанавливается на "Удалено", если в течение первого
        `n_init` кадров.
    max_age : int
- максимальное количество последовательных промахов до того, как состояние трека изменится.
        установите значение "Удалено".
    функция: Опционально[ndarray]
        Вектор признаков обнаружения, из которого исходит этот трек. Если нет,
то этот признак добавляется в кэш "признаков".

    Атрибуты
    ----------
    среднее значение : ndarray
        Вектор среднего значения распределения начального состояния.
    ковариация : ndarray
        Ковариационная матрица распределения начальных состояний.
    track_id : int
        Уникальный идентификатор трека.
    количество просмотров : общее количество обновлений измерений.

    возраст : общее количество кадров с момента первого появления.

    time_since_update :
общее количество кадров с момента последнего обновления измерений.
    состояние : TrackState
        Текущее состояние трассы.
    объекты : список[ndarray]
        Кэш объектов. При каждом обновлении результатов измерений в этот список добавляется соответствующий объект
        вектор.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Получите текущее положение в формате ограничивающего прямоугольника "(верхний левый угол x, верхний левый угол y,
ширина, высота)".

        Возвращается
        -------
        ndarray
            Ограничивающий прямоугольник.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Получить текущее положение в формате ограничивающей рамки `(minx, miny, maxx,
        max y)`.

        Возвращается
        -------
        ndarray
            Ограничивающий прямоугольник.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Распространить государственное распределение в текущем временном шаге с помощью
        Кальман шаг прогнозирования фильтр.

        Параметры
        ----------
        kf : kalman_filter.Фильтр Калмана
            Фильтр Калмана.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Выполните шаг обновления измерений фильтра Калмана и обновите функцию
        кэш.

        Параметры
        ----------
        kf : kalman_filter.Фильтр Калмана
            Фильтр Калмана.
        обнаружение : Detection
            Связанное обнаружение.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Отметьте этот трек как пропущенный (на текущем временном шаге связь отсутствует).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Возвращает значение True, если этот трек является предварительным (неподтвержденным).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Возвращает значение True, если этот трек подтвержден."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Возвращает значение True, если этот трек неактивен и должен быть удален."""
        return self.state == TrackState.Deleted