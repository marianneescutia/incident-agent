import unittest

import pandas as pd

from utils.config import TEST_DATA_PATH, TRAIN_DATA_PATH


class DataIntegrityTests(unittest.TestCase):
    def test_train_and_test_ids_do_not_overlap(self):
        train_ids = set(pd.read_csv(TRAIN_DATA_PATH)["id"])
        test_ids = set(pd.read_csv(TEST_DATA_PATH)["id"])
        self.assertFalse(train_ids & test_ids)

    def test_held_out_set_is_not_empty(self):
        self.assertGreater(len(pd.read_csv(TEST_DATA_PATH)), 0)

    def test_split_preserves_all_source_rows(self):
        train = pd.read_csv(TRAIN_DATA_PATH)
        test = pd.read_csv(TEST_DATA_PATH)
        self.assertEqual(len(train) + len(test), 50)


if __name__ == "__main__":
    unittest.main()
