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
    p.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV/TSV (TSV default) with a column named 'sequence'",
    )
    p.add_argument(
        "--seq_col",
        type=str,
        default="sequence",
        help="Column name containing sequences",
    )
    p.add_argument("--delimiter", type=str, default="\t")
    p.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to sei.pth",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument(
        "--out_npy",
        type=str,
        default="sei_preds.npy",
        help="Output .npy file (N, 21907)",
    )
    p.add_argument(
        "--out_seq_classes",
        type=str,
        default="sei_preds_seq_classes.npy",
        help="Output .npy file (N, x) where x is # of seq classes (40 to 61)",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default="sei_preds_summary.csv",
        help="Optional CSV with row index + mean/max of outputs",
    )
    p.add_argument(
        "--no_csv",
        action="store_true",
        help="Do not write summary CSV",
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input_csv, delimiter=args.delimiter)
    if args.seq_col not in df.columns:
        raise ValueError(
            f"Missing column '{args.seq_col}' in {args.input_csv}"
        )

    seqs = df[args.seq_col].tolist()
    dataset = SeqDataset(seqs, seq_len=4096)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Sei(
        sequence_length=4096,
        n_genomic_features=21907,
    )

    state = load_state_dict_flexible(
        args.pretrained,
        map_location="cpu",
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing:
        print(
            "WARNING: missing keys (showing up to 20):",
            missing[:20],
        )

    if unexpected:
        print(
            "WARNING: unexpected keys (showing up to 20):",
            unexpected[:20],
        )

    model.eval()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.to(device)

    num_sequences = len(dataset)
    num_outputs = 21907

    # Expected shape: (61, 21907)
    projvec_targets = np.load(
        "model/projvec_targets.npy"
    ).astype(np.float32, copy=False)

    if projvec_targets.ndim != 2:
        raise ValueError(
            "projvec_targets.npy must be a 2D matrix, "
            f"but found shape {projvec_targets.shape}"
        )

    if projvec_targets.shape[1] != num_outputs:
        raise ValueError(
            "Unexpected projection matrix shape. "
            f"Expected (*, {num_outputs}), "
            f"found {projvec_targets.shape}"
        )

    num_seq_classes = projvec_targets.shape[0]

    # Disk-backed arrays. These do not need to fit in RAM.
    preds = np.lib.format.open_memmap(
        args.out_npy,
        mode="w+",
        dtype=np.float32,
        shape=(num_sequences, num_outputs),
    )

    seq_class_preds = np.lib.format.open_memmap(
        args.out_seq_classes,
        mode="w+",
        dtype=np.float32,
        shape=(num_sequences, num_seq_classes),
    )

    if not args.no_csv:
        pred_means = np.empty(
            num_sequences,
            dtype=np.float32,
        )
        pred_maxes = np.empty(
            num_sequences,
            dtype=np.float32,
        )

    row_start = 0

    with torch.inference_mode():
        for x in tqdm(loader, desc="Predicting"):
            x = x.to(
                device,
                non_blocking=True,
            )  # (B, 4, 4096)

            y = model(x)  # (B, 21907)

            batch_preds = (
                y.detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            row_end = row_start + batch_preds.shape[0]

            # Write the full predictions directly to disk.
            preds[row_start:row_end] = batch_preds

            # Project only the current batch to sequence classes.
            seq_class_preds[row_start:row_end] = (
                batch_preds @ projvec_targets.T
            ).astype(np.float32, copy=False)

            if not args.no_csv:
                pred_means[row_start:row_end] = (
                    batch_preds.mean(axis=1)
                )
                pred_maxes[row_start:row_end] = (
                    batch_preds.max(axis=1)
                )

            row_start = row_end

    print("Predictions complete. Flushing outputs to disk...")

    preds.flush()
    seq_class_preds.flush()

    if not args.no_csv:
        summary = pd.DataFrame({
            "row": np.arange(num_sequences),
            "pred_mean": pred_means,
            "pred_max": pred_maxes,
        })
        summary.to_csv(args.out_csv, index=False)

    print(
        f"Saved predictions: {args.out_npy} "
        f"shape={preds.shape}"
    )

    print(
        f"Saved sequence class predictions: "
        f"{args.out_seq_classes} "
        f"shape={seq_class_preds.shape}"
    )

    if not args.no_csv:
        print(f"Saved summary: {args.out_csv}")


if __name__ == "__main__":
    main()