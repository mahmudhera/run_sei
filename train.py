import argparse
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import VariantDataset
from model_wrapper import VariantEffectModel
from utils import correlation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--delimiter", type=str, default="\t")
    # ref seq, alt seq, ref activity, alt activity column names
    parser.add_argument("--ref_col", type=str, default="ref_seq")
    parser.add_argument("--alt_col", type=str, default="alt_seq")
    parser.add_argument("--ref_act_col", type=str, default="ref_activity")
    parser.add_argument("--alt_act_col", type=str, default="alt_activity")

    parser.add_argument("--pretrained", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)

    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)

    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--freeze_backbone", action="store_true")

    return parser.parse_args()


def get_optimizer(name, model, lr_head, lr_backbone):
    backbone_params = []
    head_params = []

    for name_p, p in model.named_parameters():
        if "backbone" in name_p:
            backbone_params.append(p)
        else:
            head_params.append(p)

    if name == "adam":
        return torch.optim.Adam(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head},
            ]
        )
    elif name == "sgd":
        return torch.optim.SGD(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params, "lr": lr_head},
            ],
            momentum=0.9,
        )
    else:
        raise ValueError("Unsupported optimizer")


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    loss_fn = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for ref, alt, y in loader:
            ref, alt, y = ref.to(device), alt.to(device), y.to(device)

            out = model(ref, alt)
            loss = loss_fn(out, y)

            total_loss += loss.item() * y.size(0)

            preds.extend(out.cpu().numpy())
            targets.extend(y.cpu().numpy())

    corr = correlation(targets, preds)
    return total_loss / len(loader.dataset), corr


def main():
    args = parse_args()

    df = pd.read_csv(args.data_csv, sep=args.delimiter)

    # only keep required columns, then rename to expected names for dataset
    df = df[[args.ref_col, args.alt_col, args.ref_act_col, args.alt_act_col]]
    df.columns = ["ref_seq", "alt_seq", "ref_activity", "alt_activity"]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_loader = DataLoader(VariantDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(VariantDataset(val_df), batch_size=args.batch_size)
    test_loader = DataLoader(VariantDataset(test_df), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariantEffectModel(
        args.pretrained,
        hidden_dim=args.hidden_dim,
        freeze_backbone=args.freeze_backbone,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = get_optimizer(
        args.optimizer, model, args.lr_head, args.lr_backbone
    )

    loss_fn = nn.MSELoss()

    best_val_corr = -1

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for ref, alt, y in tqdm(train_loader):
            ref, alt, y = ref.to(device), alt.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(ref, alt)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_corr = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Corr: {val_corr:.4f}")
        print("-" * 40)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), "best_model.pt")

    print("Evaluating on test set...")
    model.load_state_dict(torch.load("best_model.pt"))
    test_loss, test_corr = evaluate(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Corr: {test_corr:.4f}")


if __name__ == "__main__":
    main()