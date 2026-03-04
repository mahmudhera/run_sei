import numpy as np
import torch
from torch.utils.data import Dataset


NUC_TO_IDX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def one_hot_encode(seq, seq_len=4096):
    """
    Center a short sequence inside 4096bp and one-hot encode.
    Padding with zeros.
    """
    arr = np.zeros((4, seq_len), dtype=np.float32)

    seq = seq.upper()
    start = (seq_len - len(seq)) // 2

    for i, base in enumerate(seq):
        if base in NUC_TO_IDX:
            arr[NUC_TO_IDX[base], start + i] = 1.0

    return arr


class VariantDataset(Dataset):
    def __init__(self, dataframe):
        """
        dataframe columns:
        ref_seq, alt_seq, ref_activity, alt_activity
        """
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ref = one_hot_encode(row.ref_seq)
        alt = one_hot_encode(row.alt_seq)

        target = row.alt_activity - row.ref_activity

        return (
            torch.tensor(ref),
            torch.tensor(alt),
            torch.tensor(target, dtype=torch.float32),
        )