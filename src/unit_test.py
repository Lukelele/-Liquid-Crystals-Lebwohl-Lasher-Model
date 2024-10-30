import numpy as np
import pytest
from LebwohlLasher import one_energy, all_energy, get_order, MC_step

@pytest.fixture
def sample_array():
    return np.array([[0, 1], [1, 1]])

@pytest.fixture
def empty_array():
    return np.array([])


class TestAllEnergy:
    def test_total_energy(self, sample_array):
        total_energy = all_energy(sample_array, 2)
        expected = sum(one_energy(sample_array, i, j, 2) 
                      for i in range(2) for j in range(2))
        assert total_energy == pytest.approx(expected, rel=1e-6)

    def test_empty_array(self, empty_array):
        assert all_energy(empty_array, 0) == 0.0


class TestGetOrder:
    def test_perfect_order(self):
        arr = np.zeros((2, 2))
        order = get_order(arr, 2)
        assert order == pytest.approx(1.0, rel=1e-6)

    def test_random_order(self):
        np.random.seed(42)
        arr = np.random.uniform(0, 2*np.pi, (3, 3))
        order = get_order(arr, 3)
        assert 0 <= order <= 1

    def test_empty_array(self, empty_array):
        assert get_order(empty_array, 0) == 0.0

class TestMCStep:
    def test_zero_temperature(self, sample_array):
        np.random.seed(42)
        ratio = MC_step(sample_array.copy(), 0.0, 2)
        assert 0 <= ratio <= 1

    def test_high_temperature(self, sample_array):
        np.random.seed(42)
        ratio = MC_step(sample_array.copy(), 2.0, 2)
        assert 0 <= ratio <= 1
