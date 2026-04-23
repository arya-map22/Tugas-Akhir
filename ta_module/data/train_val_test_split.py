import math

from torch import Tensor


def get_train_val_test_split(
    mortality_matrix: Tensor,
    train_split: float,
    val_split: float,
    test_split: float,
) -> tuple[Tensor, Tensor, Tensor]:
    assert train_split > 0
    assert val_split >= 0
    assert test_split >= 0
    assert math.isclose(train_split + val_split + test_split, 1.0)
    assert train_split > val_split and train_split > test_split
    assert mortality_matrix.dim() == 2

    n = mortality_matrix.shape[0]

    train_size = int(train_split * n)
    val_size = int(val_split * n)
    test_size = int(test_split * n)

    remainder = n - (train_size + val_size + test_size)
    if remainder > 0:
        if val_size == 0 and val_split > 0:
            val_size += 1
            remainder -= 1

        if test_size == 0 and test_split > 0:
            test_size += 1
            remainder -= 1

        train_size += remainder
        remainder -= remainder

    assert n == (train_size + val_size + test_size)
    assert remainder == 0

    train_ind = range(train_size)
    val_ind = range(train_size, train_size + val_size)
    test_ind = range(train_size + val_size, n)

    return (
        mortality_matrix[train_ind, :],
        mortality_matrix[val_ind, :],
        mortality_matrix[test_ind, :],
    )
