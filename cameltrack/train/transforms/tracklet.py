from typing import Optional

import numpy as np
import logging

import pandas as pd

from .transform import Transform

log = logging.getLogger(__name__)


class MaxAge(Transform):
    """
    Limit the age of the tracklet to a maximum value.
    """

    def __init__(self, max_age: int = 200):
        super().__init__()
        assert max_age > 0, "'max_age' must be greater than 0."
        self.max_age = max_age

    def __call__(self, df, dets=None):
        if df.empty:
            return df
        return df[df['age'] < self.max_age + 1], dets


class MaxNumObs(Transform):
    """
    Limit the number of observations in the tracklet to a maximum value.
    """

    def __init__(self, max_num_obs: int = 100):
        super().__init__()
        assert max_num_obs > 0, "'max_num_obs' must be greater than 0."
        self.max_num_obs = max_num_obs

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets
        return df.groupby('track_id', group_keys=False).apply(lambda x: x.head(self.max_num_obs)), dets  # should be ordered age ascending


class DropoutFeatures(Transform):
    """
    Randomly drop some features from the detections.
    """

    def __init__(self, p_drop_app: float = 0.2, p_drop_bbox: float = 0.2, p_drop_kps: float = 0.2):
        """
        :param p_drop_app: Base probability of dropping appearance features.
        :param p_drop_spatial: Base probability of dropping spatial features.
        """
        super().__init__()
        assert 0 <= p_drop_app <= 1, "'p_drop_app' must be in the range [0, 1]."
        assert 0 <= p_drop_bbox <= 1, "'p_drop_bbox' must be in the range [0, 1]."
        assert 0 <= p_drop_kps <= 1, "'p_drop_kps' must be in the range [0, 1]."
        self.p_drop_app = p_drop_app
        self.p_drop_bbox = p_drop_bbox
        self.p_drop_kps = p_drop_kps
        self.rng = np.random.default_rng()  # Initialize random generator

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        # Generate masking feature for each track_id and ensure that max 1 feature is dropped
        unique_ids = df['track_id'].unique()
        drop_app = self.rng.random(size=len(unique_ids)) < self.p_drop_app
        drop_bbox = self.rng.random(size=len(unique_ids)) < self.p_drop_bbox
        drop_kps = self.rng.random(size=len(unique_ids)) < self.p_drop_kps

        # Ensure at most one feature is dropped per track_id
        all_dropped = drop_app & drop_bbox & drop_kps
        # Generate random number between 0 and 1 for three-way choice
        random_choice = self.rng.random(size=len(unique_ids))
        drop_app[all_dropped] = random_choice[all_dropped] < 1 / 3
        drop_bbox[all_dropped] = (random_choice[all_dropped] >= 1 / 3) & (random_choice[all_dropped] < 2 / 3)
        drop_kps[all_dropped] = random_choice[all_dropped] >= 2 / 3

        drop_app_ids = unique_ids[drop_app]
        drop_bbox_ids = unique_ids[drop_bbox]
        drop_kps_ids = unique_ids[drop_kps]

        df["drop_app"] = df["track_id"].isin(drop_app_ids)
        dets["drop_app"] = dets["track_id"].isin(drop_app_ids)
        df["drop_bbox"] = df["track_id"].isin(drop_bbox_ids)
        dets["drop_bbox"] = dets["track_id"].isin(drop_bbox_ids)
        df["drop_kps"] = df["track_id"].isin(drop_kps_ids)
        dets["drop_kps"] = dets["track_id"].isin(drop_kps_ids)

        return df, dets


class DropoutSporadic(Transform):
    """
    Randomly drop some detections using either uniform or gaussian (based on the 'age') distribution.
    """

    def __init__(self, p_drop: float = 0.2, sigma: float = 50.0, method: str = 'uniform'):
        """
        :param p_drop: Base probability of dropping a detection.
        :param sigma: Decay parameter for the gaussian distribution.
        :param method: Choose 'uniform' or 'gaussian' for dropoff based on age.
        """
        super().__init__()
        assert 0 <= p_drop <= 1, "'p_drop' must be in the range [0, 1]."
        self.p_drop = p_drop
        self.sigma = sigma
        assert method in ['uniform', 'gaussian'], f"Unknown method: {method}."
        self.method = method

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        if self.method == 'uniform':
            # Use a simple uniform probability for dropping
            p_drop_age = np.full(len(df), self.p_drop)
        elif self.method == 'gaussian':
            # Compute the probability drop based on the 'age' column using gaussian decay
            age = df['age'].to_numpy()
            p_drop_age = self.p_drop * np.exp(- age**2 / (2*self.sigma**2))

        # Create a random mask, where we keep only rows with lower dropout probability
        keep_mask = self.rng.uniform(size=len(df)) > p_drop_age
        return df[keep_mask], dets


class DropoutStructured(Transform):
    def __init__(self):
        super().__init__()
        # TODO
        ...


class DropoutOccluded(Transform):
    def __init__(self, p_drop: float = 0.1, min_iou: float = 0.33):
        super().__init__()
        self.p_drop = p_drop
        self.min_iou = min_iou

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        occlusion_iou = df['occlusions'].apply(lambda x: max([tup[1] for tup in x], default=0))
        occlusions = occlusion_iou > self.min_iou
        drop_mask = self.rng.uniform(size=len(df)) < self.p_drop
        df = df[~occlusions & drop_mask]
        return df, dets


class SwapSporadic(Transform):
    """
    Randomly swap two detections from the tracklet using either uniform or gaussian (based on the 'age') distribution.
    """

    def __init__(self, p_swap: float = 0.1, sigma: float = 50.0, method: str = 'uniform', not_on_most_recent_image: bool = False):
        """
        :param p_swap: Base probability of swapping two detections.
        :param sigma: Decay parameter for the gaussian distribution.
        :param method: Choose 'uniform' or 'gaussian' for swap probability based on age.
        """
        super().__init__()
        assert 0 <= p_swap <= 1, "'p_swap' must be in the range [0, 1]."
        self.p_swap = p_swap
        self.sigma = sigma
        assert method in ['uniform', 'gaussian'], f"Unknown method: {method}."
        self.method = method
        self.not_on_most_recent_image = not_on_most_recent_image

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        if 'swapped' not in df.columns:
            df["swapped"] = False

        # Determine the probability of swapping based on the selected method
        if self.method == 'uniform':
            p_swap_age = np.full(len(df), self.p_swap)
        elif self.method == 'gaussian':
            age = df['age'].to_numpy()
            p_swap_age = self.p_swap * np.exp(- age**2 / (2*self.sigma**2))

        # Non-swapped indices and their corresponding swap probabilities
        non_swapped_idx = df.index[~df["swapped"]]
        swap_probabilities = p_swap_age[df.index.get_indexer(non_swapped_idx)]

        # Generate a random number for each candidate and pick those where it's less than their swap probability
        random_values = self.rng.uniform(size=len(non_swapped_idx))
        swap_candidates = non_swapped_idx[random_values < swap_probabilities]

        min_age_by_track_id = df.groupby('track_id')['age'].min()

        for idx in swap_candidates:
            if self.not_on_most_recent_image and df.loc[idx, 'age'] == min_age_by_track_id[df.loc[idx, 'track_id']]:
                continue
            # Find potential swap partners with the same age and not yet swapped
            swap_idx = df[(df.loc[idx, 'age'] == df['age']) & (~df['swapped']) & (df.index != idx)].index
            if len(swap_idx):
                swap_idx = self.rng.choice(swap_idx)
                # Swap the 'track_id' between the two detections
                df.loc[idx, 'track_id'], df.loc[swap_idx, 'track_id'] = df.loc[swap_idx, 'track_id'], df.loc[
                    idx, 'track_id']
                df.loc[idx, 'swapped'], df.loc[swap_idx, 'swapped'] = True, True

        return df, dets


class SwapOccluded(Transform):
    """
    Randomly swap occluded detections within each others with a probability of p_swap.
    """

    def __init__(self, p_swap: float = 0.5, min_iou: float = 0.33, not_on_most_recent_image: bool = False):
        super().__init__()
        assert 0 <= p_swap <= 1, "'p_swap' must be in the range [0, 1]."
        assert 0 <= min_iou <= 1, "'min_iou' must be in the range [0, 1]."
        self.p_swap = p_swap
        self.min_iou = min_iou
        self.not_on_most_recent_image = not_on_most_recent_image

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        if 'swapped' not in df.columns:
            df["swapped"] = False

        occlusions = df['occlusions'].apply(lambda x: [tup[0] for tup in x if tup[1] > self.min_iou])
        swap_candidates = df.index[occlusions.apply(len) > 0 & ~df['swapped']]
        swap_candidates = self.rng.choice(swap_candidates, size=int(len(swap_candidates) * self.p_swap), replace=False)

        min_age_by_track_id = df.groupby('track_id')['age'].min()

        for idx in swap_candidates:
            if self.not_on_most_recent_image and df.loc[idx, 'age'] == min_age_by_track_id[df.loc[idx, 'track_id']]:
                continue
                
            swap_idx_candidates = df.index[~df["swapped"] & df.index.isin(occlusions[idx])]
            if len(swap_idx_candidates):
                swap_idx = self.rng.choice(swap_idx_candidates)
                df.loc[idx, 'track_id'], df.loc[swap_idx, 'track_id'] = df.loc[swap_idx, 'track_id'], df.loc[
                    idx, 'track_id']
                df.loc[idx, 'swapped'], df.loc[swap_idx, 'swapped'] = True, True
        return df, dets


class DropDets(Transform):
    def __init__(self, p_drop=0.01):
        super().__init__()
        self.p_drop = p_drop

    def __call__(self, tracks, dets=None):
        drop_det = self.rng.random(size=len(dets)) < self.p_drop
        dets = dets[~drop_det]
        return tracks, dets


class DropEntireTracks(Transform):
    def __init__(self, p_drop=0.01):
        super().__init__()
        self.p_drop = p_drop

    def __call__(self, tracks, dets=None):
        unique_track_ids = tracks['track_id'].unique()
        drop_track = self.rng.random(size=len(unique_track_ids)) < self.p_drop
        for track_id, drop in zip(unique_track_ids, drop_track):
            if drop:
                tracks = tracks[tracks['track_id'] != track_id]

        return tracks, dets


class SwapStructured(Transform):
    def __init__(self):
        super().__init__()
        # TODO
        ...


class SwapRandomDetections(Transform):
    """
    Randomly swap two detections from the tracklet, respecting previous augmentations.
    """

    def __init__(self, p_swap: float = 0.1, max_swap_prop: float = 0.3, p_track: float = 1.0,
                 fix_person_id: bool = False, not_on_most_recent_image: bool = False):
        super().__init__()
        self.p_swap = p_swap
        self.max_swap_prop = max_swap_prop
        self.not_on_most_recent_image = not_on_most_recent_image
        self.fix_person_id = fix_person_id
        self.p_track = p_track
        assert 0 <= self.p_swap <= 1.0, "'p_swap' must be in the range [0, 1.0]."
        assert 0 <= self.max_swap_prop <= 1.0, "'max_swap_prop' must be in the range [0, 1.0]."

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        track_dfs = []

        for track_id, track_df in df.groupby("track_id"):
            if not (self.rng.random() < self.p_track):
                track_dfs.append(track_df)
                continue
            # Ensure 'da_swapped' column exists
            if 'da_swapped' not in track_df.columns:
                track_df['da_swapped'] = False
            # Generate a mask for swapping, excluding already swapped detections
            swap_mask = (self.rng.uniform(size=len(track_df)) < self.p_swap) & ~track_df['da_swapped']

            # Ensure that no more than max_swap_prop of the detections are swapped
            if swap_mask.sum() / len(track_df) > self.max_swap_prop:
                swap_idx = self.rng.choice(swap_mask.index[swap_mask], size=int(self.max_swap_prop * len(track_df)),
                                           replace=False)
                swap_mask = swap_mask.index.isin(swap_idx)

            # For the swap_mask that are True, we randomly select another detection to swap from the same 'image_id' in the 'video_df' if available
            swap_df = track_df[swap_mask].copy()
            for idx, row in swap_df.iterrows():
                image_id = row['image_id']
                swap_candidates = df[(df['image_id'] == image_id) & (df['track_id'] != track_id)].index
                if len(swap_candidates) > 1:  # Ensure there is at least one other eligible detection to swap with
                    swap_idx = self.rng.choice(swap_candidates[swap_candidates != idx])
                    swap_df.loc[idx] = df.loc[swap_idx]

            if self.not_on_most_recent_image:
                if isinstance(swap_mask, pd.Series):
                    # Index of smallest image_id:
                    min_image_id_idx = track_df['image_id'].idxmax()  # most recent = highest image id
                    # min_image_id_idx = track_df['image_id'].idxmin()  # most recent = highest image id
                    swap_mask[min_image_id_idx] = False
                else:
                    swap_mask[-1] = False
                    # swap_mask[0] = False

            # Update the original dataframe with the swapped values
            track_df.loc[swap_mask] = swap_df

            # Mark the newly swapped detections
            track_df.loc[swap_mask, 'da_swapped'] = True
            track_df.loc[swap_mask, 'track_id'] = track_id
            if self.fix_person_id:
                assert len(track_df.loc[~swap_mask, 'person_id'].unique()) == 1
                track_df.loc[swap_mask, 'person_id'] = track_df.loc[~swap_mask, 'person_id'].unique()[0]
            track_dfs.append(track_df)
        # for track_df in track_dfs:
        #     assert not track_df.loc[track_df['age'].idxmin()]['da_swapped']
        df = pd.concat(track_dfs)
        return df, dets


class SwapRandomDetectionsWithoutIdUpdate(Transform):
    """
    Randomly swap two detections from the tracklet, respecting previous augmentations.
    """

    def __init__(self, p_swap: float = 0.1, max_swap_prop: float = 0.3, fix_person_id: bool = False, not_on_most_recent_image: bool = False):
        super().__init__()
        self.p_swap = p_swap
        self.max_swap_prop = max_swap_prop
        self.not_on_most_recent_image = not_on_most_recent_image
        self.fix_person_id = fix_person_id
        assert 0 <= self.p_swap <= 1.0, "'p_swap' must be in the range [0, 1.0]."
        assert 0 <= self.max_swap_prop <= 1.0, "'max_swap_prop' must be in the range [0, 1.0]."

    def __call__(self, df, dets=None):
        if df.empty:
            return df, dets

        track_dfs = []

        for track_id, track_df in df.groupby("track_id"):
            # Ensure 'da_swapped' column exists
            if 'da_swapped' not in track_df.columns:
                track_df['da_swapped'] = False
            # Generate a mask for swapping, excluding already swapped detections
            swap_mask = (self.rng.uniform(size=len(track_df)) < self.p_swap) & ~track_df['da_swapped']

            # Ensure that no more than max_swap_prop of the detections are swapped
            if swap_mask.sum() / len(track_df) > self.max_swap_prop:
                swap_idx = self.rng.choice(swap_mask.index[swap_mask], size=int(self.max_swap_prop * len(track_df)),
                                           replace=False)
                swap_mask = swap_mask.index.isin(swap_idx)

            # For the swap_mask that are True, we randomly select another detection to swap from the same 'image_id' in the 'video_df' if available
            swap_df = track_df[swap_mask].copy()
            for idx, row in swap_df.iterrows():
                image_id = row['image_id']
                swap_candidates = df[(df['image_id'] == image_id) & (df['track_id'] != track_id)].index
                if len(swap_candidates) > 1:  # Ensure there is at least one other eligible detection to swap with
                    swap_idx = self.rng.choice(swap_candidates[swap_candidates != idx])
                    swap_df.loc[idx] = df.loc[swap_idx]

            # if self.not_on_most_recent_image:
            #     if isinstance(swap_mask, pd.Series):
            #         # Index of smallest image_id:
            #         min_image_id_idx = track_df['image_id'].idxmax()  # most recent = highest image id
            #         swap_mask[min_image_id_idx] = False
            #     else:
            #         swap_mask[0] = False

            # Update the original dataframe with the swapped values
            track_df.loc[swap_mask] = swap_df

            # Mark the newly swapped detections
            # assert isinstance(swap_mask, pd.Series)
            track_df.loc[swap_mask, 'da_swapped'] = True
            track_df.loc[swap_mask, 'track_id'] = track_id
            # if self.fix_person_id:
            #     assert len(track_df.loc[~swap_mask, 'person_id'].unique()) == 0
            #     track_df.loc[swap_mask, 'person_id'] = track_df.loc[~swap_mask, 'person_id'].unique()[0]
            track_dfs.append(track_df)
        # for track_df in track_dfs:
        #     assert not track_df.loc[track_df['age'].idxmin()]['da_swapped']
        df = pd.concat(track_dfs)
        return df, dets
