from collections import defaultdict

import numpy as np

from src import embedding_utils, general_utils, geometry_utils


def calculate_embedding_multiple(embedder, crops, distributed_count=16, return_mean=True):
    if distributed_count:
        distributed_crops = general_utils.get_distributed_items(crops, n=distributed_count)
    else:
        distributed_crops = crops

    vectors = embedder.get_embeddings(distributed_crops)
    if vectors is None:
        return None
    if return_mean:
        return np.mean(vectors, axis=0)
    return vectors


class CrossCameraMatcher:
    def __init__(self, embedder, not_from_other_camera_masks):
        self.embedder = embedder
        self.embedding_size = embedder.embedding_dim
        self.not_from_other_camera_masks = not_from_other_camera_masks

        self.strong_crops_per_ids_source = {}
        self.weak_crops_per_ids_source = {}
        self.embeddings_per_id = {}
        self.embedding_histories_query = defaultdict(list)
        self.source_gallery_track_ids = []
        self.source_gallery_embeddings = np.zeros(0)
        self.query_track_is_candidate = {}
        self.best_matches_query = defaultdict(dict)
        self.chosen_matches_query = {}

    def get_best_matches(self):
        return {
            track_id: sorted(matches_by_source_id.values(), key=lambda match: match["global_score"], reverse=True)
            for track_id, matches_by_source_id in self.best_matches_query.items()
            if matches_by_source_id
        }

    def get_chosen_matches(self):
        return {
            track_id: [match_data]
            for track_id, match_data in self.chosen_matches_query.items()
            if match_data["global_score"] > 0.0
        }

    def process_query_embeddings(self, tracks, frame):
        for track in tracks:
            track_id = int(track[4])
            if not self.query_track_is_candidate.get(track_id, False):
                self.query_track_is_candidate[track_id] = self._check_if_query_track_is_candidate(track)

        query_crops = []
        query_track_ids = []

        for track in tracks:
            track_id = int(track[4])
            if not self.query_track_is_candidate[track_id]:
                continue

            x1, y1, x2, y2 = map(int, track[:4])
            query_track_ids.append(track_id)
            query_crops.append(geometry_utils.get_shrunk_crop(frame, x1, y1, x2, y2, scale=0.8))

        all_visible_query_embeddings = calculate_embedding_multiple(self.embedder,
                                                                    query_crops,
                                                                    distributed_count=None,
                                                                    return_mean=False)
        if all_visible_query_embeddings is None:
            return

        for index, vector in enumerate(all_visible_query_embeddings):
            track_id = query_track_ids[index]
            self.embedding_histories_query[track_id].append(vector)

    def query_camera_track_is_relevant(self, track_id):
        return self.query_track_is_candidate.get(track_id, True)

    def _check_if_query_track_is_candidate(self, track):
        x1, _, x2, y2 = map(int, track[:4])
        bottom_center = (int((x1 + x2) / 2), y2)
        is_inside_excluded_area = any(
            geometry_utils.point_inside_polygon(bottom_center, mask)
            for mask in self.not_from_other_camera_masks)
        return not is_inside_excluded_area

    def append_source_camera_crop(self, track_id, crop, is_strong_crop):
        if is_strong_crop:
            self.strong_crops_per_ids_source.setdefault(track_id, []).append(crop)
        else:
            self.weak_crops_per_ids_source.setdefault(track_id, []).append(crop)

    def discard_source_camera_track(self, track_id):
        gallery_needs_refresh = track_id in self.embeddings_per_id

        self.strong_crops_per_ids_source.pop(track_id, None)
        self.weak_crops_per_ids_source.pop(track_id, None)
        self.embeddings_per_id.pop(track_id, None)

        # for matches_by_source_id in self.best_matches_query.values():
        #     matches_by_source_id.pop(track_id, None)

        if gallery_needs_refresh:
            self.refresh_source_camera_gallery()

    def discard_expired_source_records(self, current_seconds, source_exit_seconds, ttl_seconds):
        expired_source_track_ids = [
            track_id
            for track_id, exit_seconds in source_exit_seconds.items()
            if current_seconds - exit_seconds > ttl_seconds
        ]
        if not expired_source_track_ids:
            return []

        for track_id in expired_source_track_ids:
            source_exit_seconds.pop(track_id, None)
            self.strong_crops_per_ids_source.pop(track_id, None)
            self.weak_crops_per_ids_source.pop(track_id, None)
            self.embeddings_per_id.pop(track_id, None)
            for matches_by_source_id in self.best_matches_query.values():
                matches_by_source_id.pop(track_id, None)

        self.refresh_source_camera_gallery()
        return expired_source_track_ids

    def record_embeddings(self, track_id):
        crops = self.strong_crops_per_ids_source.get(track_id, []) or self.weak_crops_per_ids_source.get(track_id, [])
        if not crops:
            return False

        embedding = calculate_embedding_multiple(self.embedder, crops)
        if embedding is None:
            return False

        self.embeddings_per_id[track_id] = embedding
        return True

    def refresh_source_camera_gallery(self):
        self.source_gallery_embeddings = np.zeros((len(self.embeddings_per_id), self.embedding_size), dtype="float64")
        self.source_gallery_track_ids.clear()

        for index, source_track_id in enumerate(sorted(self.embeddings_per_id.keys())):
            self.source_gallery_embeddings[index] = self.embeddings_per_id[source_track_id]
            self.source_gallery_track_ids.append(source_track_id)

    def check_matches(self, track_id, crossed_seconds, source_exit_seconds):

        if not self.embedding_histories_query[track_id]:
            return

        query_embedding = np.mean(self.embedding_histories_query[track_id], axis=0)
        if self.source_gallery_embeddings.size == 0 or not self.source_gallery_track_ids:
            return

        closest_indices, embedding_scores = embedding_utils.find_closest_embeddings(
            query_embedding,
            self.source_gallery_embeddings)

        if not closest_indices:
            return

        for closest_index, embedding_score in zip(closest_indices, embedding_scores):
            source_track_id = self.source_gallery_track_ids[closest_index]
            is_strong, source_draw_crop = self._best_crop_for_source_camera_track(source_track_id)

            elapsed_seconds = crossed_seconds - source_exit_seconds[source_track_id]
            if elapsed_seconds < 15.0:
                continue

            match_data = self.best_matches_query[track_id].get(source_track_id)
            if match_data is None:
                match_data = {
                    "embedding_score": embedding_score,
                    "source_draw_crop": source_draw_crop,
                    "source_track_id": source_track_id,
                    "elapsed_seconds": elapsed_seconds,
                }
                self.best_matches_query[track_id][source_track_id] = match_data
            else:
                match_data["embedding_score"] = embedding_score
                match_data["elapsed_seconds"] = elapsed_seconds

            match_data["is_strong"] = is_strong

            if not is_strong:
                match_data["embedding_score"] *= 1.1

            if match_data["embedding_score"] < 0.35:
                match_data["embedding_score"] = 0.0
            match_data["global_score"] = match_data["embedding_score"]

        self._update_chosen_match(track_id)


    def _update_chosen_match(self, track_id):
        matches_by_source_id = self.best_matches_query.get(track_id)
        if not matches_by_source_id:
            self.chosen_matches_query.pop(track_id, None)
            return

        sorted_matches = sorted(
            matches_by_source_id.values(),
            key=lambda match: match["global_score"],
            reverse=True,
        )
        strongest_match = sorted_matches[0]
        if strongest_match["global_score"] <= 0.0:
            return

        current_match = self.chosen_matches_query.get(track_id)
        if (
            current_match is None
            or strongest_match["global_score"] > current_match["global_score"]
        ):
            self.chosen_matches_query[track_id] = dict(strongest_match)

    def _best_crop_for_source_camera_track(self, track_id):
        strong_crops = self.strong_crops_per_ids_source.get(track_id, [])
        if strong_crops:
            strong_crops.sort(key=lambda crop: crop.shape[1])
            return True, strong_crops[-1]

        weak_crops = self.weak_crops_per_ids_source.get(track_id, [])
        weak_crops.sort(key=lambda crop: crop.shape[1])
        return False, weak_crops[-1]
