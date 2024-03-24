# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Работа с состояниями треков, есть три состояния:
    Tentative - новый созданый трек, который еще не подтвержден
    cofirmed - подтвержденный трек
    deleted - треки котоырй нужно удалить из набора активных треков
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Одиночный трек цели с пространством состояний `(x, y, a, h)` и связанными
    скоростями, где `(x, y)` - центр ограничивающей рамки, `a` - соотношение сторон,
    и `h` - высота.

    Parameters
    mean - Средний вектор начального распределения состояния.
    covariance - Ковариационная матрица начального распределения состояния.
    track_id - Уникальный идентификатор трека.
    n_init - Количество последовательных обнаружений, прежде чем трек будет подтвержден. Состояние трека устанавливается на `Deleted`, если пропуск происходит в первых `n_init` кадрах.
    max_age - Максимальное количество последовательных пропусков, прежде чем состояние трека установится на `Deleted`.
    feature - Вектор признаков обнаружения, из которого происходит этот трек. Если не None, этот признак добавляется в кэш `features`.

    Средний вектор начального распределения состояния: Это просто набор чисел, который представляет ожидаемое начальное 
    состояние объекта. Например, если мы отслеживаем движущийся объект на плоскости, средний вектор может содержать координаты объекта
    и его скорости по осям x и y.

    Ковариационная матрица начального распределения состояния: Это матрица, которая описывает степень неопределенности 
    или разброс значений в среднем векторе. Она показывает, насколько мы уверены в каждом измерении среднего вектора. 
    Например, если у нас есть хорошая оценка координаты объекта по оси x, но неопределенная оценка его координаты по оси y, 
    ковариационная матрица будет отражать эту неопределенность.

    Вместе с средним вектором начального распределения, ковариационная матрица помогает алгоритму фильтра Калмана 
    оценить текущее состояние объекта на основе измерений и предыдущих состояний.

    hits - счетчик успешных обновлений состояния объектов
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
        '''
        (x_center, y_center, aspect_ratio, height).
        '''
        temp = self.mean[:4].copy()
        temp[2] *= temp[3]
        temp[:2] -= temp[2:] / 2
        return temp

    def to_tlbr(self):
        temp = self.to_tlwh()
        temp[2:] = temp[:2] + temp[2:]
        return temp

    def predict(self, kf):
        """
        Распространяет распределение состояния до текущего временного шага, используя шаг прогнозирования фильтра Калмана.
        Функция predict в контексте фильтра Калмана используется для прогнозирования следующего состояния системы на основе текущего состояния 
        и модели движения.
        kf - фильтр

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """
        Функция update в классе, который использует фильтр Калмана, выполняет шаг обновления измерений и обновляет кэш признаков

        Parameters
        ----------
        Kf - фильтр калмана
        """

        #Вызов метода update фильтра Калмана, который обновляет среднее значение (mean) и ковариационную матрицу (covariance) объекта 
        #на основе текущего состояния, ковариационной матрицы и преобразованного вектора измерений (detection.to_xyah()).
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """
        Если объект Tentative то меняем на Deleted
        Если время с момента обновления больше чем _max_age то он тоже становится Deleted.
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted