import torch
import numpy as np
import logging

log = logging.getLogger(__name__)


class Detection:
    def __init__(self, image_id, features, tracklab_id, frame_idx):
        for k, v in features.items():
            setattr(self, k, v)
        self.tracklab_id = tracklab_id
        self.image_id = image_id
        self.frame_idx = torch.tensor(frame_idx)
        self.similarity_with_tracklet = None
        self.similarities = None


class Tracklet(object):
    # MOT benchmark requires positive:
    count = 1  # FIXME not thread safe

    def __init__(self, detection, max_gallery_size, first_tracks=False):
        self.last_detection = detection
        # self.token = detection.token
        self.detections = [detection]
        self.state = "init" if not first_tracks else "active"
        self.id = Tracklet.count
        Tracklet.count += 1

        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_wo_hits = 0

        self.max_gallery_size = max_gallery_size

    def forward(self):
        self.age += 1
        self.time_wo_hits += 1
        if self.time_wo_hits > 1:
            self.hit_streak = 0

    def update(self, detection):
        self.detections.append(detection)
        self.detections = self.detections[-self.max_gallery_size:]
        # tracklet management
        self.hits += 1
        self.hit_streak += 1
        self.time_wo_hits = 0
        self.last_detection = detection

    def padded_features(self, name, size):
        features = torch.stack([getattr(det, name) for det in reversed(self.detections)])
        if features.shape[0] < size:
            features = torch.cat(
                [features,
                 torch.zeros(size - features.shape[0], *features.shape[1:], device=features.device) + float('nan')]
            )
        return features

    def __str__(self):
        return (f"Tracklet(id={self.id}, state={self.state}, age={self.age}, "
                f"hits={self.hits}, hit_streak={self.hit_streak}, "
                f"time_wo_hits={self.time_wo_hits}, "
                f"num_detections={len(self.detections)})")


@torch.no_grad()
class CAMELTrack(object):
    def __init__(
            self,
            camel,
            min_hits,
            max_wo_hits,
            max_gallery_size=50,
            min_init_conf=0.4,
            max_tracklet_memory_size=None,
            *args,
            **kwargs,
    ):
        self.camel = camel.eval()
        self.min_hits = min_hits
        self.max_wo_hits = max_wo_hits
        self.max_gallery_size = max_gallery_size
        self.tracklets = []
        self.frame_count = 0
        self.max_tracklet_memory_size = self.camel.transformer.max_track_ids if hasattr(self.camel.transformer, "max_track_ids") else None
        self.min_init_conf = min_init_conf
        if self.max_tracklet_memory_size is not None:
            log.info(f"DDSORT initialised with max_tracklet_memory_size={self.max_tracklet_memory_size} since EncoderWithIdTokens is used.")

    def update(self, features, tracklab_ids, image_id):
        """
        Rough overview of the update method:
        1. forward each tracklet, update {age, time_wo_hits, hit_streak}
        2. perform data association between existing tracklets and new detections
        3. update all track with matched det: update detections list, last_detection, hits, hit_streak, time_wo_hits
        4. init track with unmatched dets
        5. update state of all tracklets: init ( hit_streak < min_hits) -> active -> lost (time_wo_hits < max_wo_hits) -> dead
        6. update track list by removing dead tracklets
        7. return all active tracks
        """
        self.frame_count += 1

        # build detections from features
        detections = []
        for i in range(len(tracklab_ids)):
            features_i = {k: v[0, i] for k, v in features.items()}
            detections.append(
                Detection(
                    image_id,
                    features_i,
                    tracklab_ids[i],
                    frame_idx=self.frame_count - 1
                )
            )

        # advance state of tracklets
        for track in self.tracklets:
            track.forward()

        # associate detections to tracklets
        (
            matched,
            unmatched_trks,
            unmatched_dets,
            td_sim_matrix,
        ) = self.associate_dets_to_trks(self.tracklets, detections)

        # assert each track and det index is present exactly once in matched or unmatched_trks or unmatched_dets
        # Ensure that the track indices are all accounted for, and no duplicates in matched
        all_trk_indices = [m[0] for m in matched.tolist()] + unmatched_trks
        assert len(all_trk_indices) == len(self.tracklets)  # Ensure all indices are present
        assert len(set(all_trk_indices)) == len(all_trk_indices)  # Ensure no duplicates

        # Ensure that the detection indices are all accounted for, and no duplicates in matched
        all_det_indices = [m[1] for m in matched.tolist()] + unmatched_dets
        assert len(all_det_indices) == len(detections)  # Ensure all indices are present
        assert len(set(all_det_indices)) == len(all_det_indices)  # Ensure no duplicates

        # update matched tracklets with assigned detections
        for m in matched:
            tracklet = self.tracklets[m[0]]
            detection = detections[m[1]]
            tracklet.update(detection)
            similarity = td_sim_matrix[m[0], m[1]]
            detection.similarity_with_tracklet = similarity
            detection.similarities = td_sim_matrix[:len(self.tracklets), m[1]]

        # create and initialise new tracklets for unmatched detections
        first_tracks = len(self.tracklets) == 0
        for i in unmatched_dets:
            if detections[i].bbox_conf < self.min_init_conf:
                continue
            trk = Tracklet(detections[i], self.max_gallery_size, first_tracks=first_tracks)
            self.tracklets.append(trk)

        # handle tracklets outputs and cleaning
        actives = []
        for trk in self.tracklets:
            # get active tracklets
            self.update_state(trk)
            if trk.state == "active":
                actives.append(
                    {
                        "tracklab_id": trk.last_detection.tracklab_id.item(),
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S", trk.last_detection.similarity_with_tracklet.cpu().numpy()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in
                                  enumerate(trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.camel.sim_threshold,
                        }
                    }
                )
            # for first frames, also return tracklets in init state but not confirm them as active
            elif trk.state == "init" and self.frame_count <= self.min_hits:
                actives.append(
                    {
                        "tracklab_id": trk.last_detection.tracklab_id.item(),
                        "track_id": trk.id,  # id is computed from a counter
                        "hits": trk.hits,
                        "age": trk.age,
                        "matched_with": ("S",
                                         trk.last_detection.similarity_with_tracklet.cpu().numpy()) if trk.last_detection.similarity_with_tracklet is not None else None,
                        "time_since_update": trk.time_wo_hits,
                        "state": trk.state,
                        "costs": {
                            "S": {self.tracklets[j].id: sim for j, sim in
                                  enumerate(
                                      trk.last_detection.similarities.cpu().numpy())} if trk.last_detection.similarities is not None else None,
                            "St": self.camel.sim_threshold,
                        }
                    }
                )
        self.tracklets = [trk for trk in self.tracklets if trk.state != "dead"]
        if self.max_tracklet_memory_size is not None and len(self.tracklets) > self.max_tracklet_memory_size:
            self.tracklets = self.tracklets[-self.max_tracklet_memory_size:]
        return actives

    def associate_dets_to_trks(self, tracklets, detections):
        if len(tracklets) == 0:
            return (
                np.empty((0, 2)),
                np.empty((0,)).tolist(),
                np.arange(len(detections)).tolist(),
                np.empty((0,)),
            )
        if len(detections) == 0:
            return (
                np.empty((0, 2)),
                np.arange(len(self.tracklets)).tolist(),
                np.empty((0,)).tolist(),
                np.empty((0,)),
            )

        batch = self.build_camel_batch(tracklets, detections, self.camel.device)
        association_matrix, association_result, td_sim_matrix = self.camel.predict_step(batch, self.frame_count)
        matched = association_result[0]["matched_td_indices"]
        unmatched_trks = association_result[0]["unmatched_trackers"]
        unmatched_dets = association_result[0]["unmatched_detections"]
        return matched, unmatched_trks, unmatched_dets, td_sim_matrix.squeeze(0)

    # def matched_td_indices(self, tracklet_detection_matrix):
    #     # Convert the input matrix to a PyTorch tensor
    #     matrix_tensor = torch.tensor(tracklet_detection_matrix, dtype=torch.float32)
    #
    #     # Find matched pairs (tracklet, detection)
    #     matched_pairs = []
    #     while matrix_tensor.sum() > 0:
    #         max_val, max_idx = torch.max(matrix_tensor, dim=1)
    #         tracklet_idx = torch.argmax(max_val)
    #         detection_idx = max_idx[tracklet_idx].item()
    #
    #         if max_val[tracklet_idx] > 0:
    #             matched_pairs.append((tracklet_idx, detection_idx))
    #             matrix_tensor[tracklet_idx, :] = 0
    #             matrix_tensor[:, detection_idx] = 0
    #
    #     # Find unmatched detections and tracklets
    #     unmatched_dets = torch.nonzero(matrix_tensor.sum(dim=0)).squeeze().tolist()
    #     unmatched_trks = torch.nonzero(matrix_tensor.sum(dim=1)).squeeze().tolist()
    #
    #     return matched_pairs, unmatched_dets, unmatched_trks

    def build_camel_batch(self, tracklets, detections, device):  # TODO UGLY - refactor
        T_max = np.array([len(t.detections) for t in tracklets]).max()
        batch = {
            'image_id': detections[0].image_id,  # int
            'det_feats': {
                'visibility_scores': torch.stack([det.visibility_scores for det in detections]).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 7]
                'embeddings': torch.stack([det.embeddings for det in detections]).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 7*D]
                'index': torch.IntTensor([det.tracklab_id for det in detections]).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1]
                'bbox_conf': torch.stack([det.bbox_conf for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 1]
                'bbox_ltwh': torch.stack([det.bbox_ltwh for det in detections]).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 4]
                'keypoints_xyc': torch.stack([det.keypoints_xyc for det in detections]).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 17, 3]
                'age': torch.zeros(len(detections), device=device).unsqueeze(1).unsqueeze(1).unsqueeze(0),  # [B, N, 1, 1]
                'im_width': torch.stack([det.im_width for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 1]
                'im_height': torch.stack([det.im_height for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device=device),  # [1, N, 1, 1]
            },
            'det_masks': torch.ones((1, len(detections), 1), device=device, dtype=torch.bool),  # [1, N, 1]
            'track_feats': {
                'visibility_scores': torch.stack([t.padded_features("visibility_scores", T_max) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T, 7]
                'embeddings': torch.stack([t.padded_features("embeddings", T_max) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T, 7*D]
                'index': torch.stack([t.padded_features("tracklab_id", T_max) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T]
                'bbox_conf': torch.stack([t.padded_features("bbox_conf", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device),  # [1, N, T, 1]
                'bbox_ltwh': torch.stack([t.padded_features("bbox_ltwh", T_max) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T, 4]
                'keypoints_xyc': torch.stack([t.padded_features("keypoints_xyc", T_max) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T, 17, 3]
                'age': self.frame_count - 1 - torch.stack([t.padded_features("frame_idx", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device),  # [1, N, T, 1]
                'im_width': torch.stack([t.padded_features("im_width", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device),  # [1, N, T, 1]
                'im_height': torch.stack([t.padded_features("im_height", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device),  # [1, N, T, 1]
            },
            'track_masks': torch.stack([torch.cat([torch.ones(len(t.detections), dtype=torch.bool), torch.zeros(T_max - len(t.detections), dtype=torch.bool)]) for t in tracklets]).unsqueeze(0).to(device=device),  # [1, N, T]
        }

        if hasattr(detections[0], "target"):
            batch["det_targets"] = torch.stack([det.target for det in detections]).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device=device)
            batch["track_targets"] = torch.stack([t.padded_features("target", T_max) for t in tracklets]).unsqueeze(2).unsqueeze(0).to(device=device)
        return batch

    def update_state(self, tracklet):
        s = tracklet.state
        if s == "init":
            if tracklet.hit_streak >= self.min_hits:
                new_state = "active"
            elif tracklet.time_wo_hits >= 1:
                new_state = "dead"
            else:
                new_state = "init"
        elif s == "active":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "lost":
            if tracklet.time_wo_hits == 0:
                new_state = "active"
            elif tracklet.time_wo_hits < self.max_wo_hits:
                new_state = "lost"
            else:
                new_state = "dead"
        elif s == "dead":
            new_state = "dead"
        else:
            raise ValueError(f"tracklet state is in undefined state {s}.")
        tracklet.state = new_state
