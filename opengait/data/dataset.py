import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr, is_bool_list, is_int_list

class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.dataset_name = data_cfg['dataset_name']
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache'] if 'cache' in data_cfg.keys() else False
        if 'd-gait' in self.dataset_name.lower():
            self.seqs_info = [[seq[0], f"{seq[0]}-{seq[1]}", *seq[2:]] for seq in self.seqs_info]
        if training:
            if 'self_supervised' in data_cfg.keys() and data_cfg['self_supervised']:
                self.seqs_info = [[f"{seq[0]}-{seq[1]}-{seq[2]}", *seq[1:]] for seq in self.seqs_info]
            self.seqs_info = [[f"{self.dataset_name}-{seq[0]}", *seq[1:]] for seq in self.seqs_info]
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __load_one_modality(self, paths):
        if not is_int_list(self.data_in_use):
            # When specified as a list of int, arbitrary order can be specified by the user
            paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(data)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __loader__(self, paths):
        """Load one or several aligned modalities.

        The original contract is ``paths=[frame_path, ...]``. Multimodal
        models may pass ``paths=[[map_paths...], [pose_paths...]]``.
        """
        if paths and isinstance(paths[0], (list, tuple)):
            return [self.__load_one_modality(modality_paths) for modality_paths in paths]
        return self.__load_one_modality(paths)

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        configured_roots = data_config['dataset_root']
        dataset_roots = (
            [str(root) for root in configured_roots]
            if isinstance(configured_roots, (list, tuple))
            else [str(configured_roots)]
        )
        if not dataset_roots:
            raise ValueError('dataset_root must contain at least one path.')
        missing_roots = [root for root in dataset_roots if not osp.isdir(root)]
        if missing_roots:
            raise FileNotFoundError(
                'Dataset root does not exist: %s' % ', '.join(missing_roots)
            )
        dataset_root = dataset_roots[0]
        # [n] bool mask or [m] explicit indices.
        self.data_in_use = data_config.get('data_in_use')

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s], Len: %s' %
                                 (pid_list[0], pid_list[1], pid_list[-1], len(pid_list)))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_aligned_paths(seq_info):
            """Return frame paths for one root or aligned modality roots."""
            modality_paths = []
            reference_names = None
            for root in dataset_roots:
                seq_path = osp.join(root, *seq_info)
                if not osp.isdir(seq_path):
                    return []
                names = sorted(
                    name for name in os.listdir(seq_path)
                    if name.lower().endswith('.pkl')
                )
                if reference_names is None:
                    reference_names = names
                elif names != reference_names:
                    raise ValueError(
                        'Multimodal sequence files are not aligned for %s' % '-'.join(seq_info)
                    )
                modality_paths.append([osp.join(seq_path, name) for name in names])
            if not modality_paths or reference_names == []:
                return []
            return modality_paths if len(dataset_roots) > 1 else modality_paths[0]

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = get_aligned_paths(seq_info)
                        if seq_dirs != []:
                            if self.data_in_use is not None:
                                if is_bool_list(self.data_in_use): # self.data_in_use is a bool mask list
                                    if len(dataset_roots) > 1:
                                        seq_dirs = [
                                            [path for path, use_bl in zip(modality, self.data_in_use) if use_bl]
                                            for modality in seq_dirs
                                        ]
                                    else:
                                        seq_dirs = [dir for dir, use_bl in zip(
                                            seq_dirs, self.data_in_use) if use_bl]
                                else:   # self.data_in_use is a list of index
                                    if len(dataset_roots) > 1:
                                        seq_dirs = [
                                            [modality[idx] for idx in self.data_in_use if idx < len(modality)]
                                            for modality in seq_dirs
                                        ]
                                    else:
                                        seq_dirs = [seq_dirs[idx] for idx in self.data_in_use if idx < len(seq_dirs)]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)
