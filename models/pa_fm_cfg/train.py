"""
Train PA-FM-CFG (Performance-Aware Flow Matching with CFG)
============================================================
Architecture params (hidden_dim, n_blocks, p_uncond, guidance_scale, sampling_steps)
inherited from FM-CFG HPO best. Only PA-specific params are tuned here.

Usage:
    python models/pa_fm_cfg/train.py
    python models/pa_fm_cfg/train.py --lambda_perf 0.3 --ode_steps 5
"""

import os, sys, time, copy, json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(MODEL_DIR))
sys.path.insert(0, ROOT_DIR)

from shared.data_utils import get_dataloaders
from models.pa_fm_cfg.model import PerformanceAwareFlowMatching

CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
RESULT_DIR = os.path.join(MODEL_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args):
    print(f"[Device] {DEVICE}")
    train_loader, val_loader, _, info = get_dataloaders(batch_size=args.batch_size, use_merged=True)
    print(f"Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']}")

    # Load FM-CFG best params for architecture
    fm_best_path = os.path.join(ROOT_DIR, "models", "fm_cfg", "results",
                                "hpo_best_fm_cfg.json")
    with open(fm_best_path) as f:
        fm_params = json.load(f)["best_params"]
    print(f"FM-CFG base: {fm_params}")

    model = PerformanceAwareFlowMatching(
        geom_dim=21, cond_dim=3,
        hidden_dim=fm_params["hidden_dim"], n_blocks=fm_params["n_blocks"],
        p_uncond=fm_params["p_uncond"],
        guidance_scale=fm_params.get("guidance_scale", 2.0),
        sampling_steps=fm_params.get("sampling_steps", 100),
        dropout=fm_params.get("dropout", 0.1),
        lambda_perf=args.lambda_perf,
        warmup_epochs=args.warmup_epochs,
        ramp_epochs=args.ramp_epochs,
        ode_steps=args.ode_steps,
        t_min=args.t_min,
        device=str(DEVICE),
    ).to(DEVICE)

    trainable = model.get_trainable_params()
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    lr = fm_params.get("lr", 1e-4)
    optimizer = Adam(trainable, lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_flow": [], "train_perf": [], "val_flow": [], "lambda": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_flow, ep_perf, n = 0.0, 0.0, 0
        for batch in train_loader:
            g, l = batch["geometry"].to(DEVICE), batch["label"].to(DEVICE)
            loss, info_d = model.training_loss(g, l, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            bs = g.shape[0]
            ep_flow += info_d["flow"] * bs
            ep_perf += info_d["perf"] * bs
            n += bs
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                g, l = batch["geometry"].to(DEVICE), batch["label"].to(DEVICE)
                _, v_info = model.training_loss(g, l, epoch=epoch)
                vl += v_info["flow"] * g.shape[0]
                vn += g.shape[0]
        val_flow = vl / vn

        history["train_flow"].append(ep_flow / n)
        history["train_perf"].append(ep_perf / n)
        history["val_flow"].append(val_flow)
        history["lambda"].append(info_d["lambda"])

        if val_flow < best_val:
            best_val = val_flow
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4d}/{args.epochs}  flow={ep_flow/n:.6f}/{val_flow:.6f}  "
                  f"perf={ep_perf/n:.4f}  lam={info_d['lambda']:.3f}  ({time.time()-t0:.1f}s)")

        if no_improve >= args.patience and epoch >= args.min_epochs:
            print(f"  early stop @ epoch {epoch}")
            break

    ckpt_path = os.path.join(CKPT_DIR, "pa_fm_cfg.pth")
    torch.save(best_state, ckpt_path)
    print(f"Saved -> {ckpt_path}")

    with open(os.path.join(RESULT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    import types
    args = types.SimpleNamespace(
        epochs=400, batch_size=256, patience=80, min_epochs=150,
        lambda_perf=0.2, warmup_epochs=80, ramp_epochs=50,
        ode_steps=5, t_min=0.2,
    )
    train(args)
