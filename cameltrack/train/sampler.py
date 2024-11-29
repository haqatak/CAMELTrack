import logging
import pickle
import random
from collections import OrderedDict
from itertools import islice
from math import ceil
from zipfile import ZipFile

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from rich.progress import track
from torch.utils.data import Sampler

log = logging.getLogger(__name__)

class ConcatSampler(Sampler):
    def __init__(self, samplers, batch_size, num_samples):
        super().__init__(None)
        self.samplers = samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.rng = np.random.default_rng()


    def __len__(self):
        return sum(len(s) for s in self.samplers) // self.batch_size

    def sample_generator(self):
        samplers = [iter(sampler) for sampler in self.samplers]
        probs = np.array([len(s) for s in self.samplers], dtype=np.float64)
        probs = probs / probs.sum()
        indexes = list(range(len(samplers)))
        while samplers:
            key = self.rng.choice(range(len(samplers)), p=probs)
            current_sampler = samplers[key]
            try:
                item = next(current_sampler)
            except StopIteration:
                del samplers[key]
                del indexes[key]
                probs = list(probs)
                del probs[key]
                probs = np.array(probs, dtype=np.float64) / np.array(probs, dtype=np.float64).sum()
                continue
            assert len(
                item) == 1, "Always set the batch size to 1 when using ConcatSampler"
            item = item[0]
            index = indexes[key]
            yield [(index, *s) for s in item]

    def __iter__(self):
        yield from batched(self.sample_generator(), self.batch_size)


class CAMELSampler(Sampler):
    def __init__(self, dataset, batch_size=128, num_samples=8, samples_per_video=1,
                 fill_samples=False, **kwargs):
        super().__init__(dataset)
        self.rng = np.random.default_rng()
        self.dataset = dataset
        assert hasattr(dataset, "samples"), "You should define the samples"
        self.samples = self.dataset.samples
        self.df_samples = pd.DataFrame(self.samples)
        assert len(np.unique([x["global_track_id"] for x in self.samples])) == len(self.samples), "All tracklets should have different IDs"
        self.track_ids = np.sort([x["global_track_id"] for x in self.samples])
        self.samples_per_video = samples_per_video
        self.unique_video_ids = np.sort(np.unique([x["video_id"] for x in self.samples]))
        self.video_ids = np.repeat(self.unique_video_ids, self.samples_per_video)
        # self.image_ids = np.sort(np.unique([x["image_id"] for x in self.samples]))
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.fill_samples = fill_samples

    def __len__(self):
        return ceil(len(self.video_ids) / self.batch_size)

    def sample_generator(self):
        """Generator of tuples with sample_idx and random image_id"""
        random_video_ids = self.rng.choice(self.video_ids, len(self.video_ids), replace=False)
        samples = self.df_samples  # pd.DataFrame(self.samples)
        for video_id in random_video_ids:
            video_samples = samples[samples["video_id"] == video_id]
            possible_image_ids = np.unique(np.concatenate(np.array(video_samples["image_id"])))
            image_id = self.rng.choice(possible_image_ids)
            tracklets = []
            if self.fill_samples:
                samples_left = self.num_samples
                rest_video_ids = self.unique_video_ids.copy()
                while samples_left > 0:
                    tracklets.extend([(video_id, s, image_id) for i, s in zip(range(samples_left), video_samples.index)])
                    samples_left -= len(video_samples)
                    rest_video_ids = rest_video_ids[rest_video_ids != video_id]
                    if len(rest_video_ids) == 0:
                        tracklets.extend([(video_id, -1, image_id) for _ in range(samples_left)])
                        samples_left = 0
                        continue
                    video_id = self.rng.choice(rest_video_ids)
                    video_samples = samples[samples["video_id"] == video_id]
                    possible_image_ids = np.unique(np.concatenate(np.array(video_samples["image_id"])))
                    image_id = self.rng.choice(possible_image_ids)
            else:
                if len(video_samples) > self.num_samples:
                    sample_range = random.sample(range(self.num_samples), self.num_samples)
                else:
                    sample_range = range(self.num_samples)
                for i in sample_range:
                    try:
                        sample_index = video_samples.index[i]
                    except IndexError:
                        sample_index = -1
                    tracklets.append((video_id, sample_index, image_id))
            yield tracklets

        if len(random_video_ids) % self.batch_size != 0:
            tracklets = []
            for _ in range(self.batch_size - (len(random_video_ids) % self.batch_size)):
                for _ in range(self.num_samples):
                    tracklets.append((-1, -1, -1))  # yield -1, -1
            yield tracklets

    def __iter__(self):
        yield from batched(self.sample_generator(), self.batch_size)


class OcclusionSampler(CAMELSampler):

    def __init__(self, dataset, batch_size=128, num_samples=8, samples_per_video=1,
                 window_size=5, min_iou=0.25, base_probability=1,
                 prob_occlusion_sampling=1, **kwargs):
        super().__init__(dataset, batch_size, num_samples, samples_per_video, **kwargs)
        self.window = np.concatenate([np.ones(window_size), np.zeros(window_size-1)])
        self.min_iou = min_iou
        self.base_prob = base_probability
        self.prob_occlusion_sampling = prob_occlusion_sampling
        self.video_dfs = {}
        for video_id in track(np.unique(self.video_ids), "Reading videos pickles"):
            with ZipFile(self.dataset.gallery_path).open(video_id, 'r') as fp:
                self.video_dfs[video_id] = pickle.load(fp)

    def sample_generator(self):
        random_video_ids = self.rng.choice(self.video_ids, len(self.video_ids), replace=False)
        samples = self.df_samples
        next_video = None
        for video_id in random_video_ids:
            video_samples = samples[samples["video_id"] == video_id]
            video_df = self.video_dfs[video_id]
            video_df["num_occlusions"] = video_df["occlusions"].apply(lambda x: len([tup[0] for tup in x if tup[1] > self.min_iou]))
            num_occlusions = video_df.groupby("image_id")["num_occlusions"].sum()
            image_ids = num_occlusions.index
            num_occlusions = np.correlate(num_occlusions, self.window, mode="same")  # Also pick images after
            probabilities = num_occlusions + self.base_prob
            probabilities = probabilities.astype(np.float64) / probabilities.sum()
            sample_occlusions = self.rng.random() < self.prob_occlusion_sampling
            if sample_occlusions:
                image_id = self.rng.choice(image_ids, p=probabilities)
            else:
                image_id = self.rng.choice(image_ids)
            tracklets = []
            for i in range(self.num_samples):
                try:
                    sample_index = video_samples.index[i]
                except IndexError:
                    sample_index = -1
                tracklets.append((video_id, sample_index, image_id))
            yield tracklets

        if len(random_video_ids) % self.batch_size != 0:
            tracklets = []
            for _ in range(self.batch_size - (len(random_video_ids) % self.batch_size)):
                for _ in range(self.num_samples):
                    tracklets.append((-1, -1, -1))  # yield -1, -1
            yield tracklets


class GapSampler(CAMELSampler):

    def __init__(self, dataset, batch_size=128, num_samples=8, samples_per_video=1, **kwargs):
        super().__init__(dataset, batch_size, num_samples, samples_per_video, **kwargs)
        self.video_dfs = {}
        for video_id in track(np.unique(self.video_ids), "Reading videos pickles"):
            with ZipFile(self.dataset.gallery_path).open(video_id, 'r') as fp:
                self.video_dfs[video_id] = pickle.load(fp)

    def sample_generator(self):
        random_video_ids = self.rng.choice(self.video_ids, len(self.video_ids), replace=False)
        samples = self.df_samples
        for video_id in random_video_ids:
            video_samples = samples[samples["video_id"] == video_id]
            video_df = self.video_dfs[video_id]
            no_obs = video_df.groupby("image_id")["no_obs"].sum() + 1
            image_id = self.rng.choice(no_obs.index, p=(no_obs.astype(np.float64)/no_obs.sum()))
            tracklets = []
            for i in range(self.num_samples):
                try:
                    sample_index = video_samples.index[i]
                except IndexError:
                    sample_index = -1
                tracklets.append((video_id, sample_index, image_id))
            yield tracklets

        if len(random_video_ids) % self.batch_size != 0:
            tracklets = []
            for _ in range(self.batch_size - (len(random_video_ids) % self.batch_size)):
                for _ in range(self.num_samples):
                    tracklets.append((-1, -1, -1))  # yield -1, -1
            yield tracklets


class ValSampler(Sampler):
    def __init__(self, dataset, batch_size=128):
        super().__init__(dataset)
        self.dataset = dataset
        self.samples = pd.DataFrame(dataset.samples)
        self.image_ids = np.unique(self.samples.image_id)
        self.batch_size = batch_size
        self.num_samples = self.samples.groupby("image_id")["detections"].count().max()
        self.dl_batch_size = batch_size * self.num_samples

    def __len__(self):
        return (len(self.samples) + self.dl_batch_size - 1) // self.dl_batch_size

    def __iter__(self):
        for img_ids in batched(self.image_ids, self.batch_size):
            batched_samples = []
            for img_id in img_ids:
                samples = self.samples[self.samples.image_id == img_id]
                samples = samples.index
                samples = np.pad(samples, pad_width=(0, self.num_samples-len(samples)),
                                 constant_values=-1)
                batched_samples.extend(samples)
            if (len(batched_samples) % self.batch_size) != 0:
                batched_samples = np.pad(batched_samples,
                                         pad_width=(0, self.batch_size - (len(batched_samples) % self.batch_size)),
                                         constant_values=-1)
            yield batched_samples


class RandomFrameSampler(Sampler):
    def __init__(self, dataset, batch_size=128, num_samples=8):
        super().__init__(dataset)
        self.rng = np.random.default_rng(1234)
        self.dataset = dataset
        assert hasattr(dataset, "samples"), "You should define the samples"
        self.samples = self.dataset.samples
        self.image_ids = np.unique([x["image_id"] for x in self.samples])
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.dl_batch_size = batch_size * num_samples

    def __len__(self):
        return (len(self.samples) + self.dl_batch_size - 1) // self.dl_batch_size

    def __iter__(self):
        # use ordered sets and dicts to make this function deterministic when seed is set
        samples = pd.DataFrame(self.samples)
        # group set of image ids by video id
        video_id_to_image_id_set = samples.sort_values(['video_id', 'image_id']).groupby('video_id')['image_id'].apply(OrderedSet).to_dict()
        video_id_to_image_id_set = OrderedDict(sorted(video_id_to_image_id_set.items()))
        # pop image ids from video_id_to_image_id_set until none remaining
        pairs_list_for_epoch = []
        while len(video_id_to_image_id_set) > 0:
            # build one training sample
            # a training sample is a list of "num_samples" pairs of tracklet-detection
            # it correspond to an instance of a tracklet-to-detection association occurring in online tracking
            videos_to_pick_from = OrderedSet(video_id_to_image_id_set.keys())
            chosen_pairs = []
            while len(chosen_pairs) < self.num_samples and len(videos_to_pick_from) > 0:
                # pick a random video
                video_id = self.rng.choice(list(videos_to_pick_from))
                videos_to_pick_from.remove(video_id)
                # pick a random frame from that video
                image_id = self.rng.choice(list(video_id_to_image_id_set[video_id]))
                video_id_to_image_id_set[video_id].remove(image_id)
                if len(video_id_to_image_id_set[video_id]) == 0:
                    video_id_to_image_id_set.pop(video_id)
                # pick all pairs from that frame
                pairs = samples[samples['image_id'] == image_id]
                chosen_pairs.extend(list(pairs.index))

            if len(chosen_pairs) >= self.num_samples:
                chosen_pairs = chosen_pairs[:self.num_samples]  # FIXME not optimal, some pairs are lost
                pairs_list_for_epoch.extend(chosen_pairs)

        yield from batched(pairs_list_for_epoch, self.dl_batch_size)


samplers = {
    "simple": CAMELSampler,
    "occlusion": OcclusionSampler,
    "gap": GapSampler,
    "hard": NotImplemented,
    "harder": NotImplemented,
    "val": CAMELSampler,  # ValSampler,
    "random_frame": RandomFrameSampler,
}


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.

    >>> list(batched('ABCDEFG', 3))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G',)]

    On Python 3.12 and above, this is an alias for :func:`itertools.batched`.
    """
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            break
        yield batch
