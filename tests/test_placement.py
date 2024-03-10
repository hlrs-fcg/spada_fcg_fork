import unittest
import numpy as np

from spatialstencil.placement.partition import FieldPartition


class TestPlacement(unittest.TestCase):
    def test_placement(self):
        # 1 by 1
        partition_array = np.array([[0, 0], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 1)
        assert np.allclose(placement.strides[:, 1], 1)

        # 2 by 2
        partition_array = np.array([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 2)
        assert np.allclose(placement.strides[:, 1], 2)

        # 1 by 2
        partition_array = np.array([[0, 1], [0, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose( placement.strides[:, 0], 1)
        assert np.allclose(placement.strides[:, 1], 2)

        # 2 by 1
        partition_array = np.array([[0, 0], [1, 0]], dtype=np.int32)
        partition = FieldPartition(partition_array)
        placement = partition.place_interleaved()
        assert np.allclose(placement.strides[:, 0], 2)
        assert np.allclose(placement.strides[:, 1], 1)


if __name__ == '__main__':
    unittest.main()
