# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Решите задачу линейного назначения.

    Параметры
    ----------
    distance_metric : Вызываемый[Список[дорожек], Список[обнаружений], Список[int], Список[int]) -> ndarray
        Метрика расстояния содержит список дорожек и обнаружений, а также
        список из N индексов отслеживания и M индексов обнаружения. Показатель должен
        возвращает размерную матрицу затрат NxM, где элемент (i, j) представляет
собой стоимость связи между i-м треком в заданных индексах трека и
j-м обнаружением в заданных индексах обнаружения.
    max_distance : значение с плавающей точкой
        Порог стробирования. Связи со стоимостью, превышающей это значение
, не учитываются.
    дорожки : Список[дорожка.Дорожка]
        Список прогнозируемых дорожек на текущем временном шаге.
    обнаружения : Список[обнаружение.Обнаружение]
        Список обнаружений на текущем временном шаге.
    track_indices : Список[int]
        Список индексов дорожек, которые сопоставляют строки в `cost_matrix` с дорожками в
`tracks` (см. описание выше).
    detection_indices : список[int]
        Список индексов обнаружения, которые сопоставляют столбцы в `cost_matrix` с
обнаружениями в `detections` (см. описание выше).

    Возвращается
    -------
    (List[(int, int)], List[int], List[int])
        Возвращает кортеж со следующими тремя элементами:
        * Список совпадающих индексов трека и обнаружения.
        * Список несовпадающих индексов трека.
        * Список непревзойденных показателей обнаружения.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)
    indices = np.asarray(indices)
    indices = np.transpose(indices)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Запустите соответствующий каскад.

    Параметры
    ----------
    distance_metric : Вызываемый[Список[дорожек], Список[обнаружений], Список[int], Список[int]) -> ndarray
        Метрика расстояния содержит список дорожек и обнаружений, а также
        список из N индексов треков и M индексов обнаружения. Метрика должна
        возвращать размерную матрицу затрат NxM, где элемент (i, j) представляет
собой стоимость связи между i-м треком в заданных индексах треков и
j-м обнаружением в заданных индексах обнаружения.
    максимальное расстояние : плавающее значение
        Порог стробирования. Связи со стоимостью, превышающей это значение
, не учитываются.
    cascade_depth:
указывает глубину каскада, которая должна соответствовать максимальному возрасту трека.
    треки : Список[трек.Трек]
        Список прогнозируемых дорожек на текущем временном шаге.
    обнаружения : Список[обнаружение.Detection]
        Список обнаружений на текущем временном шаге.
    track_indices : Необязательный[Список[int]]
        Список индексов дорожек, который сопоставляет строки в `cost_matrix` с дорожками в
`tracks` (см. описание выше). По умолчанию используется для всех дорожек.
    detection_indices : Необязательный[Список[int]]
        Список индексов обнаружения, который сопоставляет столбцы в `cost_matrix` с
обнаружениями в `detections` (см. описание выше). По умолчанию используется значение all
        обнаружения.

    Возвращается
    -------
    (List[(int, int)], List[int], List[int])
        Возвращает кортеж со следующими тремя элементами:
        * Список совпадающих индексов трека и обнаружения.
        * Список несовпадающих индексов трека.
        * Список несовпадающих индексов обнаружения.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Отмените недопустимые записи в матрице затрат на основе
распределений состояний, полученных с помощью фильтрации Калмана.

    Параметры
    ----------
    kf : Фильтр Калмана.
    cost_matrix : ndarray
        Размерная матрица затрат NxM, где N - количество индексов отслеживания
        а M - это количество индексов обнаружения, так что запись (i, j) - это стоимость
связи между "треками[track_indices[i]]" и
"обнаружениями[detection_indices[j]]".
    треки : Список[трек.Трек]
        Список прогнозируемых дорожек на текущем временном шаге.
    обнаружения : Список[обнаружение.Обнаружение]
        Список обнаружений на текущем временном шаге.
    track_indices : Список[int]
        Список индексов дорожек, которые сопоставляют строки в `cost_matrix` с дорожками в
`tracks` (см. описание выше).
    detection_indices : список[int]
        Список индексов обнаружения, которые сопоставляют столбцы в `cost_matrix` с
обнаружениями в `detections` (см. описание выше).
    gated_cost : Необязательно[с плавающей точкой]
        Это значение задается для записей в матрице затрат, соответствующих недопустимым связям
. По умолчанию используется очень большое значение.
    only_position : Необязательно[bool]
        Если значение True, то учитывается только положение x, y в распределении состояний
        во время стробирования. По умолчанию используется значение False.

    Возвращается
    -------
    ndarray
        Возвращает измененную матрицу затрат.
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix