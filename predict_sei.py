import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data import one_hot_encode
from model.sei import Sei

from utils import load_state_dict_flexible


class SeqDataset(Dataset):
    def __init__(self, seqs, seq_len=4096):
        self.seqs = [str(s) for s in seqs]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = one_hot_encode(self.seqs[idx], seq_len=self.seq_len)  # (4, 4096)
        return torch.tensor(x, dtype=torch.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True,
                   help="CSV with a column named 'sequence'")
    p.add_argument("--seq_col", type=str, default="sequence",
                   help="Column name containing sequences")
    p.add_argument("--pretrained", type=str, required=True,
                   help="Path to sei.pth")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--out_npy", type=str, default="sei_preds.npy",
                   help="Output .npy file (N, 21907)")
    p.add_argument("--out_csv", type=str, default="sei_preds_summary.csv",
                   help="Optional CSV with row index + mean/max of outputs")
    p.add_argument("--no_csv", action="store_true",
                   help="Do not write summary CSV")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv)
    if args.seq_col not in df.columns:
        raise ValueError(f"Missing column '{args.seq_col}' in {args.input_csv}")

    seqs = df[args.seq_col].tolist()
    dataset = SeqDataset(seqs, seq_len=4096)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Sei(sequence_length=4096, n_genomic_features=21907)
    state = load_state_dict_flexible(args.pretrained, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        print("WARNING: missing keys (showing up to 20):", missing[:20])
    if unexpected:
        print("WARNING: unexpected keys (showing up to 20):", unexpected[:20])
    
    model.eval()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    preds = []
    with torch.no_grad():
        for x in tqdm(loader, desc="Predicting"):
            x = x.to(device, non_blocking=True)  # (B, 4, 4096)
            y = model(x)                         # (B, 21907)
            preds.append(y.detach().cpu().numpy().astype(np.float32))

    preds = np.concatenate(preds, axis=0)  # (N, 21907)
    np.save(args.out_npy, preds)

    if not args.no_csv:
        # lightweight summary; avoids writing huge CSV of 21k columns
        summary = pd.DataFrame({
            "row": np.arange(preds.shape[0]),
            "pred_mean": preds.mean(axis=1),
            "pred_max": preds.max(axis=1),
        })
        summary.to_csv(args.out_csv, index=False)

    print(f"Saved predictions: {args.out_npy}  shape={preds.shape}")
    if not args.no_csv:
        print(f"Saved summary: {args.out_csv}")


if __name__ == "__main__":
    main()