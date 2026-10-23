from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "opengait"))

from data.sampler import ClinicalClassBalancedSampler  # noqa: E402


class _ToyClinicalDataset:
    def __init__(self):
        self.label_set = [f"pid-{index}" for index in range(9)]
        classes = ["negative"] * 3 + ["neutral"] * 3 + ["positive"] * 3
        self.label_list = []
        self.types_list = []
        self.indices_dict = {}
        index = 0
        for pid, status in zip(self.label_set, classes):
            self.indices_dict[pid] = [index, index + 1]
            self.label_list.extend([pid, pid])
            self.types_list.extend([status, status])
            index += 2


class ClinicalSamplerTests(unittest.TestCase):
    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.broadcast", side_effect=lambda tensor, src=0: tensor)
    def test_each_batch_balances_the_three_clinical_classes(self, *_):
        dataset = _ToyClinicalDataset()
        sampler = ClinicalClassBalancedSampler(
            dataset=dataset,
            batch_size=[6, 1],
            batch_shuffle=False,
        )
        sampled_indices = next(iter(sampler))
        sampled_types = [dataset.types_list[index] for index in sampled_indices]
        self.assertEqual(sampled_types.count("negative"), 2)
        self.assertEqual(sampled_types.count("neutral"), 2)
        self.assertEqual(sampled_types.count("positive"), 2)


if __name__ == "__main__":
    unittest.main()
