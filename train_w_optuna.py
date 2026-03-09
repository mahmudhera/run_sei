import argparse
import copy
import os

import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import VariantDataset
from model_wrapper import VariantEffectModel
from utils import correlation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_csv", type=str, required=True)
    parser.add_argument("--delimiter", type=str, default="\t")

    # input column names
    parser.add_argument("--ref_col", type=str, default="ref_seq")
    parser.add_argument("--alt_col", type=str, default="alt_seq")
    parser.add_argument("--ref_act_col", type=str, default="ref_activity")
    parser.add_argument("--alt_act_col", type=str, default="alt_activity")

    parser.add_argument("--pretrained", type=str, required=True)

    # default / fallback training params
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--freeze_backbone", action="store_true")

    # optuna args
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="variant_effect_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # search space controls
    parser.add_argument("--search_batch_size", action="store_true")
    parser.add_argument("--search_hidden_dim", action="store_true")
    parser.add_argument("--search_lr_head", action="store_true")
    parser.add_argument("--search_lr_backbone", action="store_true")
    parser.add_argument("--search_optimizer", action="store_true")
    parser.add_argument("--search_freeze_backbone", action="store_true")

    # optional pruning
    parser.add_argument("--use_pruner", action="store_true")

    # output files
    parser.add_argument("--best_model_path", type=str, default="best_model.pt")
    parser.add_argument("--best_params_path", type=str, default="best_params.txt")

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(name, model, lr_head, lr_backbone):
    backbone_params = []
    head_params = []

    for name_p, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name_p:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head})

    if name == "adam":
        return torch.optim.Adam(param_groups)
    elif name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def make_loaders(train_df, val_df, test_df, batch_size, num_workers=0):
    train_loader = DataLoader(
        VariantDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        VariantDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        VariantDataset(test_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def build_model(pretrained, hidden_dim, freeze_backbone, device):
    model = VariantEffectModel(
        pretrained,
        hidden_dim=hidden_dim,
        freeze_backbone=freeze_backbone,
    )

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    return model


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for ref, alt, y in loader:
            ref = ref.to(device)
            alt = alt.to(device)
            y = y.to(device)

            out = model(ref, alt)
            loss = loss_fn(out, y)

            total_loss += loss.item() * y.size(0)
            preds.extend(out.detach().cpu().numpy().reshape(-1))
            targets.extend(y.detach().cpu().numpy().reshape(-1))

    corr = correlation(targets, preds)
    return total_loss / len(loader.dataset), corr


def train_one_epoch(model, loader, optimizer, loss_fn, device, show_progress=True):
    model.train()
    total_loss = 0.0

    iterator = tqdm(loader, leave=False) if show_progress else loader

    for ref, alt, y in iterator:
        ref = ref.to(device)
        alt = alt.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(ref, alt)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def suggest_hparams(trial, args):
    batch_size = (
        trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        if args.search_batch_size
        else args.batch_size
    )

    hidden_dim = (
        trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
        if args.search_hidden_dim
        else args.hidden_dim
    )

    lr_head = (
        trial.suggest_float("lr_head", 1e-5, 1e-2, log=True)
        if args.search_lr_head
        else args.lr_head
    )

    lr_backbone = (
        trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
        if args.search_lr_backbone
        else args.lr_backbone
    )

    optimizer_name = (
        trial.suggest_categorical("optimizer", ["adam", "sgd"])
        if args.search_optimizer
        else args.optimizer
    )

    freeze_backbone = (
        trial.suggest_categorical("freeze_backbone", [False, True])
        if args.search_freeze_backbone
        else args.freeze_backbone
    )

    return {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "optimizer": optimizer_name,
        "freeze_backbone": freeze_backbone,
    }


def train_with_params(
    params,
    args,
    train_df,
    val_df,
    device,
    trial=None,
    save_best_path=None,
    show_progress=True,
):
    train_loader, val_loader, _ = make_loaders(
        train_df,
        val_df,
        None if False else val_df,  # placeholder to satisfy function signature shape
        batch_size=params["batch_size"],
        num_workers=args.num_workers,
    )
    # ignore third returned loader
    train_loader, val_loader = train_loader, val_loader

    model = build_model(
        args.pretrained,
        hidden_dim=params["hidden_dim"],
        freeze_backbone=params["freeze_backbone"],
        device=device,
    )

    optimizer = get_optimizer(
        params["optimizer"],
        model,
        params["lr_head"],
        params["lr_backbone"],
    )

    loss_fn = nn.MSELoss()

    best_val_corr = float("-inf")
    best_val_loss = None
    best_epoch = -1
    best_state_dict = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, show_progress=show_progress
        )
        val_loss, val_corr = evaluate(model, val_loader, device)

        print(
            f"[Trial {trial.number if trial is not None else 'final'}] "
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Corr: {val_corr:.4f}"
        )

        if trial is not None:
            trial.report(val_corr, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(model.state_dict())

    if save_best_path is not None and best_state_dict is not None:
        torch.save(best_state_dict, save_best_path)

    return {
        "best_val_corr": best_val_corr,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
    }


def retrain_and_test(best_params, best_epoch, args, train_df, val_df, test_df, device):
    combined_train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    train_loader = DataLoader(
        VariantDataset(combined_train_df),
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        VariantDataset(test_df),
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.pretrained,
        hidden_dim=best_params["hidden_dim"],
        freeze_backbone=best_params["freeze_backbone"],
        device=device,
    )

    optimizer = get_optimizer(
        best_params["optimizer"],
        model,
        best_params["lr_head"],
        best_params["lr_backbone"],
    )

    loss_fn = nn.MSELoss()

    best_state_dict = None
    best_train_loss = float("inf")

    for epoch in range(best_epoch):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, show_progress=True
        )
        print(
            f"[Final retrain] Epoch {epoch + 1}/{best_epoch} | "
            f"Train Loss: {train_loss:.4f}"
        )

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state_dict = copy.deepcopy(model.state_dict())

    torch.save(best_state_dict, args.best_model_path)
    model.load_state_dict(torch.load(args.best_model_path, map_location=device))

    test_loss, test_corr = evaluate(model, test_loader, device)
    return test_loss, test_corr


def main():
    args = parse_args()
    set_seed(args.seed)

    df = pd.read_csv(args.data_csv, sep=args.delimiter)

    df = df[[args.ref_col, args.alt_col, args.ref_act_col, args.alt_act_col]]
    df.columns = ["ref_seq", "alt_seq", "ref_activity", "alt_activity"]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pruner = optuna.pruners.MedianPruner() if args.use_pruner else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True if args.storage is not None else False,
        direction="maximize",
        pruner=pruner,
    )

    def objective(trial):
        params = suggest_hparams(trial, args)

        result = train_with_params(
            params=params,
            args=args,
            train_df=train_df,
            val_df=val_df,
            device=device,
            trial=trial,
            save_best_path=None,
            show_progress=False,
        )

        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("best_val_loss", result["best_val_loss"])

        return result["best_val_corr"]

    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"  Value (Val Corr): {study.best_value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    print(f"  Best epoch: {study.best_trial.user_attrs['best_epoch']}")

    # merge searched params with defaults so final config is complete
    best_params = {
        "batch_size": study.best_trial.params.get("batch_size", args.batch_size),
        "hidden_dim": study.best_trial.params.get("hidden_dim", args.hidden_dim),
        "lr_head": study.best_trial.params.get("lr_head", args.lr_head),
        "lr_backbone": study.best_trial.params.get("lr_backbone", args.lr_backbone),
        "optimizer": study.best_trial.params.get("optimizer", args.optimizer),
        "freeze_backbone": study.best_trial.params.get(
            "freeze_backbone", args.freeze_backbone
        ),
    }
    best_epoch = study.best_trial.user_attrs["best_epoch"]

    with open(args.best_params_path, "w") as f:
        f.write(f"Best validation correlation: {study.best_value:.6f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    print("\nRetraining final model on train + val using best hyperparameters...")
    test_loss, test_corr = retrain_and_test(
        best_params=best_params,
        best_epoch=best_epoch,
        args=args,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        device=device,
    )

    print("\nFinal test results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Corr: {test_corr:.4f}")
    print(f"Best model saved to: {args.best_model_path}")
    print(f"Best params saved to: {args.best_params_path}")


if __name__ == "__main__":
    main()