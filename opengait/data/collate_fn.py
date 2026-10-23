import math
import random
import numpy as np


class CollateFn(object):
    def __init__(self, label_set, sample_config):
        self.label_set = label_set
        sample_type = sample_config['sample_type']
        sample_type = sample_type.split('_')
        self.sampler = sample_type[0]
        order_mode = sample_type[1]
        if self.sampler not in ['fixed', 'unfixed', 'all']:
            raise ValueError
        if order_mode not in ['ordered', 'unordered', 'allordered']:
            raise ValueError
        self.ordered = order_mode in ['ordered', 'allordered']
        self.allordered = order_mode == 'allordered'

        # fixed cases
        if self.sampler == 'fixed':
            self.frames_num_fixed = sample_config['frames_num_fixed']

        # unfixed cases
        if self.sampler == 'unfixed':
            self.frames_num_max = sample_config['frames_num_max']
            self.frames_num_min = sample_config['frames_num_min']

        if self.sampler != 'all' and self.ordered:
            self.frames_skip_num = sample_config['frames_skip_num']

        self.frames_all_limit = -1
        if self.sampler == 'all' and 'frames_all_limit' in sample_config:
            self.frames_all_limit = sample_config['frames_all_limit']

        self.points_in_use = sample_config.get('points_in_use')
        # Some released medical gait pickles store one complete sequence in a
        # single file ([T, ...]) instead of one file per frame.  Keep the
        # original OpenGait behaviour by default and opt into frame-level
        # sampling explicitly from the dataset config.
        self.sample_from_sequence = bool(sample_config.get('sample_from_sequence', False))
        # Some modalities describe the whole recording (for example, raw pose
        # used to calculate a sequence-level clinical prior).  Keep those
        # modalities intact while the visual streams are frame-sampled.
        self.full_sequence_modalities = tuple(
            int(index) for index in sample_config.get(
                'full_sequence_modalities', []
            )
        )

    def __call__(self, batch):
        batch_size = len(batch)
        # currently, the functionality of feature_num is not fully supported yet, it refers to 1 now. We are supposed to make our framework support multiple source of input data, such as silhouette, or skeleton.
        feature_num = len(batch[0][0])
        invalid_modalities = [
            index for index in self.full_sequence_modalities
            if index < 0 or index >= feature_num
        ]
        if invalid_modalities:
            raise ValueError(
                'full_sequence_modalities contains invalid indices: %s'
                % invalid_modalities
            )
        seqs_batch, labs_batch, typs_batch, vies_batch = [], [], [], []

        for bt in batch:
            seqs_batch.append(bt[0])
            labs_batch.append(self.label_set.index(bt[1][0]))
            typs_batch.append(bt[1][1])
            vies_batch.append(bt[1][2])

        def sample_frames(seqs, sample_index):
            sampled_fras = [[] for i in range(feature_num)]
            sample_from_sequence = (
                self.sample_from_sequence
                and len(seqs[0]) == 1
                and isinstance(seqs[0][0], np.ndarray)
                and seqs[0][0].ndim >= 1
            )
            seq_len = int(seqs[0][0].shape[0]) if sample_from_sequence else len(seqs[0])
            if sample_from_sequence:
                modality_lengths = [int(modality[0].shape[0]) for modality in seqs]
                if any(length != seq_len for length in modality_lengths):
                    raise ValueError(
                        'Aligned modalities have different frame counts: %s'
                        % modality_lengths
                    )
            if seq_len == 0:
                raise ValueError(
                    'Find no frames in sequence %s-%s-%s.'
                    % (str(labs_batch[sample_index]),
                       str(typs_batch[sample_index]),
                       str(vies_batch[sample_index]))
                )
            indices = list(range(seq_len))

            if self.sampler in ['fixed', 'unfixed']:
                if self.sampler == 'fixed':
                    frames_num = self.frames_num_fixed
                else:
                    frames_num = random.choice(
                        list(range(self.frames_num_min, self.frames_num_max+1)))
                if self.allordered:
                    replace = seq_len < frames_num
                    indices = sorted(np.random.choice(
                        indices, frames_num, replace=replace))
                elif self.ordered:
                    fs_n = frames_num + self.frames_skip_num
                    if seq_len < fs_n:
                        it = math.ceil(fs_n / seq_len)
                        seq_len = seq_len * it
                        indices = indices * it

                    start = random.choice(list(range(0, seq_len - fs_n + 1)))
                    end = start + fs_n
                    idx_lst = list(range(seq_len))
                    idx_lst = idx_lst[start:end]
                    idx_lst = sorted(np.random.choice(
                        idx_lst, frames_num, replace=False))
                    indices = [indices[i] for i in idx_lst]
                else:
                    replace = seq_len < frames_num
                    indices = np.random.choice(
                        indices, frames_num, replace=replace)

            limited_indices = (
                indices[:self.frames_all_limit]
                if self.frames_all_limit > -1 and len(indices) > self.frames_all_limit
                else indices
            )
            for i in range(feature_num):
                modality_indices = (
                    range(modality_lengths[i])
                    if sample_from_sequence and i in self.full_sequence_modalities
                    else limited_indices
                )
                for j in modality_indices:
                    frame = seqs[i][0][j] if sample_from_sequence else seqs[i][j]
                    point_cloud_index = self.points_in_use.get('pointcloud_index') if self.points_in_use else None
                    if self.points_in_use is not None and point_cloud_index is not None and i == point_cloud_index:
                        points_num = self.points_in_use.get('points_num')
                        sample_points = (random.choices(range(len(frame)), k=points_num)
                                if points_num is not None else list(range(len(frame))))
                        sampled_fras[i].append(np.asarray([frame[p] for p in sample_points]))
                    else:
                        sampled_fras[i].append(frame)
            return sampled_fras

        # f: feature_num
        # b: batch_size
        # p: batch_size_per_gpu
        # g: gpus_num
        fras_batch = [sample_frames(seqs, index)
                      for index, seqs in enumerate(seqs_batch)]  # [b, f]
        batch = [fras_batch, labs_batch, typs_batch, vies_batch, None]

        if self.sampler == "fixed":
            fras_batch = [[np.asarray(fras_batch[i][j]) for i in range(batch_size)]
                          for j in range(feature_num)]  # [f, b]
        else:
            sampled_lengths = [[len(fras_batch[i][0])
                                for i in range(batch_size)]]  # [1, p]

            def my_cat(k): return np.concatenate(
                [fras_batch[i][k] for i in range(batch_size)], 0)
            fras_batch = [[my_cat(k)] for k in range(feature_num)]  # [f, g]

            if self.full_sequence_modalities:
                batch[-1] = {
                    'sampled': np.asarray(sampled_lengths),
                    'modalities': {
                        index: np.asarray([[
                            len(fras_batch_sample[index])
                            for fras_batch_sample in batch[0]
                        ]])
                        for index in self.full_sequence_modalities
                    },
                }
            else:
                batch[-1] = np.asarray(sampled_lengths)

        batch[0] = fras_batch
        return batch
