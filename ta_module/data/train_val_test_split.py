from pandas import DataFrame


def get_train_val_test_split(
    df: DataFrame,
    train_split: float,
    val_split: float,
    test_split: float,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    assert train_split > 0
    assert val_split >= 0
    assert test_split >= 0
    assert train_split + val_split + test_split == 1
    assert train_split > val_split and train_split > test_split

    index = df.index
    n = len(index)
    train_size = int(train_split * len(index))
    val_size = int(val_split * len(index))
    test_size = int(test_split * len(index))

    remainder = n - train_size + val_size + test_size
    if remainder > 0:
        if val_size == 0 and val_split > 0:
            val_size += 1
            remainder -= 1

        if test_size == 0 and test_split > 0:
            test_size += 1
            remainder -= 1

        train_size += remainder
        remainder -= remainder

    assert n == train_size + val_size + test_size
    assert remainder == 0

    train_ind = index[:train_size]
    val_ind = index[train_size : train_size + val_size]
    test_ind = index[train_size + val_size + test_size :]

    return df.loc[train_ind, :], df.loc[val_ind, :], df.loc[test_ind, :]
