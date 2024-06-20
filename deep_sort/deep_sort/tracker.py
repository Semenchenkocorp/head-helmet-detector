# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    Это многоцелевой трекер.

    Параметры
    ----------
    метрика : nn_matching.Метрика расстояния до ближайшего соседа
        Метрика расстояния для привязки измерения к треку.
    max_age :
Максимальное количество пропущенных пропусков перед удалением трека.
    n_init :
число последовательных обнаружений до подтверждения трека. Состояние
трека устанавливается на "Удалено", если в течение первых кадров происходит пропуск
        n_init.

    Атрибуты
    ----------
    метрика : nn_матчинг.Метрика расстояния до ближайшего соседа
        Показатель расстояния, используемый для измерения связи треков.
    max_age :
максимальное количество пропущенных кадров перед удалением трека.
    n_init :
количество кадров, в течение которых трек остается на этапе инициализации.
    kf : kalman_filter.Фильтр Калмана
        Фильтр Калмана для фильтрации целевых траекторий в пространстве изображения.
    дорожки : Список[Track]
        Список активных дорожек на текущем временном шаге.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Распространяйте распределение состояний отслеживания на один временной шаг вперед.

        Эта функция должна вызываться один раз на каждом временном шаге перед `обновлением`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Выполняйте обновление измерений и управление отслеживанием.

        Параметры
        ----------
        обнаружения : Список[deep_sort.detection.Обнаружение]
            Список обнаружений на текущем временном шаге.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

      # Разделите набор дорожек на подтвержденные и неподтвержденные.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Сопоставьте подтвержденные треки, используя функции внешнего вида.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

      # Свяжите оставшиеся треки с неподтвержденными треками, используя долговую расписку.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1