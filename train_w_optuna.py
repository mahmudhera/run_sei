import argparse
import copy
import json
import random

import numpy as np
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

    # fallback defaults (also used when a parameter is not searched)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--freeze_backbone", action="store_true")

    # optuna
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--study_name", type=str, default="variant_effect_optuna")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_pruner", action="store_true")

    # search toggles
    parser.add_argument("--search_batch_size", action="store_true")
    parser.add_argument("--search_hidden_dim", action="store_true")
    parser.add_argument("--search_lr_head", action="store_true")
    parser.add_argument("--search_lr_backbone", action="store_true")
    parser.add_argument("--search_weight_decay", action="store_true")
    parser.add_argument("--search_optimizer", action="store_true")
    parser.add_argument("--search_freeze_backbone", action="store_true")

    # output
    parser.add_argument("--best_model_path", type=str, default="best_model.pt")
    parser.add_argument("--best_params_path", type=str, default="best_params.json")
    parser.add_argument("--trials_csv_path", type=str, default="optuna_trials.csv")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, batch_size, shuffle, num_workers):
    return DataLoader(
        VariantDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


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


def get_optimizer(name, model, lr_head, lr_backbone, weight_decay):
    backbone_params = []
    head_params = []

    for param_name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in param_name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay}
        )
    if head_params:
        param_groups.append(
            {"params": head_params, "lr": lr_head, "weight_decay": weight_decay}
        )

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found.")

    if name == "adam":
        return torch.optim.Adam(param_groups)
    if name == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9)

    raise ValueError(f"Unsupported optimizer: {name}")


def evaluate(model, loader, device):
    model.eval()
    preds = []
    targets = []
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for ref, alt, y in loader:
            ref = ref.to(device, non_blocking=True)
            alt = alt.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(ref, alt)
            loss = loss_fn(out, y)

            total_loss += loss.item() * y.size(0)

            preds.extend(out.detach().cpu().numpy().reshape(-1).tolist())
            targets.extend(y.detach().cpu().numpy().reshape(-1).tolist())

    corr = correlation(targets, preds)
    return total_loss / len(loader.dataset), corr


def train_one_epoch(model, loader, optimizer, device, show_progress=False):
    model.train()
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    iterator = tqdm(loader, leave=False) if show_progress else loader

    for ref, alt, y in iterator:
        ref = ref.to(device, non_blocking=True)
        alt = alt.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(ref, alt)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

    return total_loss / len(loader.dataset)


def suggest_hparams(trial, args):
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        if args.search_batch_size
        else args.batch_size,
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
        if args.search_hidden_dim
        else args.hidden_dim,
        "lr_head": trial.suggest_float("lr_head", 1e-5, 1e-2, log=True)
        if args.search_lr_head
        else args.lr_head,
        "lr_backbone": trial.suggest_float("lr_backbone", 1e-6, 1e-3, log=True)
        if args.search_lr_backbone
        else args.lr_backbone,
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        if args.search_weight_decay
        else args.weight_decay,
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd"])
        if args.search_optimizer
        else args.optimizer,
        "freeze_backbone": trial.suggest_categorical("freeze_backbone", [False, True])
        if args.search_freeze_backbone
        else args.freeze_backbone,
    }
    return params


def run_trial(params, args, train_df, val_df, device, trial=None, show_progress=False):
    train_loader = make_loader(
        train_df,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        val_df,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        pretrained=args.pretrained,
        hidden_dim=params["hidden_dim"],
        freeze_backbone=params["freeze_backbone"],
        device=device,
    )

    optimizer = get_optimizer(
        name=params["optimizer"],
        model=model,
        lr_head=params["lr_head"],
        lr_backbone=params["lr_backbone"],
        weight_decay=params["weight_decay"],
    )

    best_val_corr = float("-inf")
    best_val_loss = None
    best_epoch = None
    best_state_dict = None

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            show_progress=show_progress,
        )
        val_loss, val_corr = evaluate(model, val_loader, device)

        prefix = f"[Trial {trial.number}]" if trial is not None else "[Final]"
        print(
            f"{prefix} Epoch {epoch + 1}/{args.epochs} | "
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

    return {
        "best_val_corr": best_val_corr,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_state_dict": best_state_dict,
    }


def retrain_and_test(best_params, best_epoch, args, train_df, val_df, test_df, device):
    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    trainval_loader = make_loader(
        trainval_df,
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        test_df,
        batch_size=best_params["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(
        pretrained=args.pretrained,
        hidden_dim=best_params["hidden_dim"],
        freeze_backbone=best_params["freeze_backbone"],
        device=device,
    )

    optimizer = get_optimizer(
        name=best_params["optimizer"],
        model=model,
        lr_head=best_params["lr_head"],
        lr_backbone=best_params["lr_backbone"],
        weight_decay=best_params["weight_decay"],
    )

    best_state_dict = None
    best_train_loss = float("inf")

    for epoch in range(best_epoch):
        train_loss = train_one_epoch(
            model=model,
            loader=trainval_loader,
            optimizer=optimizer,
            device=device,
            show_progress=True,
        )

        print(
            f"[Retrain] Epoch {epoch + 1}/{best_epoch} | "
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
    df = df[[args.ref_col, args.alt_col, args.ref_act_col, args.alt_act_col]].copy()
    df.columns = ["ref_seq", "alt_seq", "ref_activity", "alt_activity"]

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if args.use_pruner
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
        direction="maximize",
        pruner=pruner,
    )

    def objective(trial):
        params = suggest_hparams(trial, args)

        result = run_trial(
            params=params,
            args=args,
            train_df=train_df,
            val_df=val_df,
            device=device,
            trial=trial,
            show_progress=False,
        )

        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("best_val_loss", float(result["best_val_loss"]))
        return result["best_val_corr"]

    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    best_params = {
        "batch_size": best_trial.params.get("batch_size", args.batch_size),
        "hidden_dim": best_trial.params.get("hidden_dim", args.hidden_dim),
        "lr_head": best_trial.params.get("lr_head", args.lr_head),
        "lr_backbone": best_trial.params.get("lr_backbone", args.lr_backbone),
        "weight_decay": best_trial.params.get("weight_decay", args.weight_decay),
        "optimizer": best_trial.params.get("optimizer", args.optimizer),
        "freeze_backbone": best_trial.params.get(
            "freeze_backbone", args.freeze_backbone
        ),
    }
    best_epoch = int(best_trial.user_attrs["best_epoch"])

    print("\nBest trial summary")
    print(f"Best validation correlation: {best_trial.value:.6f}")
    print(f"Best epoch: {best_epoch}")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    payload = {
        "best_validation_correlation": float(best_trial.value),
        "best_epoch": best_epoch,
        "best_params": best_params,
    }
    with open(args.best_params_path, "w") as f:
        json.dump(payload, f, indent=2)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(args.trials_csv_path, index=False)

    print("\nRetraining on train + val with best hyperparameters...")
    test_loss, test_corr = retrain_and_test(
        best_params=best_params,
        best_epoch=best_epoch,
        args=args,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        device=device,
    )

    print("\nFinal test results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Corr: {test_corr:.4f}")
    print(f"Best model saved to: {args.best_model_path}")
    print(f"Best params saved to: {args.best_params_path}")
    print(f"Trials CSV saved to: {args.trials_csv_path}")


if __name__ == "__main__":
    main()