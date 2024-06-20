# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Таблица для 0,95-го квантиля распределения хи-квадрат с N степенями
свободы (содержит значения для N=1, ..., 9). Взято из
функции chi2inv в MATLAB/Octave и используется в качестве порога стробирования Махаланобиса.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    Простой фильтр Калмана для отслеживания ограничивающих рамок в пространстве изображения.

    8-мерное пространство состояний

        x, y, a, h, vx, vy, va, vh

    содержит положение центра ограничивающего прямоугольника (x, y), соотношение сторон a, высоту h
и соответствующие скорости.

    Движение объекта соответствует модели с постоянной скоростью. Расположение ограничивающего прямоугольника
    (x, y, a, h) берется как прямое наблюдение за пространством состояний (линейная
    модель наблюдения).
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Создайте дорожку на основе несвязанного измерения.

        Параметры
        ----------
        измерение : ndarray
            Координаты ограничивающего прямоугольника (x, y, a, h) с указанием положения центра (x, y),
соотношения сторон a и высоты h.

        Возвращается
        -------
        (ndarray, ndarray)
            Возвращает вектор среднего значения (8-мерный) и ковариационную матрицу (8x8
-мерный) нового трека. Ненаблюдаемые скорости инициализируются
            значение равно 0.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Запустите этап прогнозирования фильтра Калмана.

        Параметры
        ----------
        среднее значение : ndarray
            8-мерный вектор среднего значения состояния объекта на предыдущем
            временном шаге.
        Ковариация : ndarray
            8x8-мерная ковариационная матрица состояния объекта на
предыдущем временном шаге.

        Возвращается
        -------
        (ndarray, ndarray)
            Возвращает вектор среднего значения и ковариационную матрицу прогнозируемого
            государство. Ненаблюдаемые скорости инициализируются как средние значения 0.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Распределение состояния проекта по измерительному пространству.

        Параметры
        ----------
        среднее значение : ndarray
            Вектор среднего значения состояния (8-мерный массив).
        Ковариация : ndarray
            Ковариационная матрица состояния (8x8-мерная).

        Возвращается
        -------
        (ndarray, ndarray)
            Возвращает прогнозируемое среднее значение и ковариационную матрицу данного состояния
            оценивать.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Выполните шаг коррекции фильтра Калмана.

        Параметры
        ----------
        среднее значение : ndarray
            Вектор среднего значения прогнозируемого состояния (8-мерный).
        Ковариация : ndarray
            Ковариационная матрица состояния (8x8-мерная).
        измерение : ndarray
            4-мерный вектор измерения (x, y, a, h), где (x, y)
            - это положение центра, a - соотношение сторон, а h - высота
ограничивающего прямоугольника.

        Возвращается
        -------
        (ndarray, ndarray)
            Возвращает скорректированное с помощью измерений распределение состояний.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Вычислите расстояние между распределением состояний и измерениями.

        Подходящее пороговое значение расстояния можно получить из `chi2inv95`. Если
значение `only_position` равно False, то распределение хи-квадрат имеет 4 степени
свободы, в противном случае - 2.

        Параметры
        ----------
        среднее значение : ndarray
            Средний вектор распределения состояний (8-мерный).
        ковариация : ndarray
            Ковариация распределения состояний (8x8-мерный).
        измерения : ndarray
            Матрица измерений Nx4, состоящая из N измерений, каждое из которых имеет
            формат (x, y, a, h), где (x, y) - центр ограничивающего прямоугольника
            положение, a - соотношение сторон, а h - высота.
        only_position : Необязательно[bool]
            Если значение равно True, вычисление расстояния выполняется относительно ограничивающего прямоугольника.
            только в центральном положении коробки.

        Возвращается
        -------
        ndarray
            Возвращает массив длиной N, где i-й элемент содержит
квадрат расстояния Махаланобиса между (средним значением, ковариацией) и
`измерениями[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha