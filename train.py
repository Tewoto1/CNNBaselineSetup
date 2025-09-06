# train.py
import os, argparse, yaml, time, copy, math
from tqdm.auto import tqdm
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.resnet_baseline import ResNetBaseline
from losses.cross_entropy import LabelSmoothingCE

# -------------------------
# Utils
# -------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def set_seed(seed):
    if seed is None: return
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def topk_correct(logits, target, ks=(1,5)):
    with torch.no_grad():
        maxk = max(ks)
        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in ks:
            res.append(correct[:k].reshape(-1).float().sum().item())
        return res

def maybe_freeze(model, freeze_up_to="none"):
    # freeze_up_to in {"none","conv1","layer1","layer2","layer3","all"}
    names = ["conv1","bn1","layer1","layer2","layer3","layer4","fc"]
    cutoff = {
        "none": -1,
        "conv1": 0,
        "layer1": 2,
        "layer2": 3,
        "layer3": 4,
        "all":   6
    }.get(freeze_up_to, -1)
    if cutoff < 0: return
    blocks = [model.net.conv1, model.net.bn1, model.net.layer1,
              model.net.layer2, model.net.layer3, model.net.layer4, model.net.fc]
    for i, m in enumerate(blocks):
        req_grad = (i > cutoff)
        for p in m.parameters():
            p.requires_grad = req_grad

def build_dataloaders(cfg):
    name = cfg["dataset"]["name"]
    img_size = cfg["dataset"]["img_size"]
    bs = cfg["dataset"]["batch_size"]
    nw = cfg["dataset"]["workers"]

    if cfg["dataset"].get("wds", False):
        # streaming via WebDataset
        from datasets.imagenet_wds import make_imagenet_wds
        train = make_imagenet_wds(cfg["dataset"]["train_urls"], "train", img_size, bs, nw)
        val   = make_imagenet_wds(cfg["dataset"]["val_urls"],   "val",   img_size, bs, nw)
        return train, val, None  # class count handled by cfg.model.num_classes
    
    if name == "food101":
        from datasets.Food101_fs import make_food101_fs
        root = cfg["dataset"]["root"]
        train_ds = make_food101_fs(root="./data", split = "train", img_size = img_size)
        val_ds   = make_food101_fs(root="./data", split = "val",   img_size = img_size)
        num_classes = 101
    else:
        raise ValueError(f"Unknown dataset {name}")

    train = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers= nw, pin_memory=True)
    val   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers= nw, pin_memory=True)
    return train, val, num_classes

def build_loss(cfg):
    lcfg = cfg["loss"]
    if lcfg["name"] == "label_smoothing_ce":
        return LabelSmoothingCE(smoothing=float(lcfg.get("smoothing", 0.1)))
    elif lcfg["name"] == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss {lcfg['name']}")

def build_optimizer(cfg, model):
    ocfg = cfg["optimizer"]
    params = []
    head_names = {"net.fc.weight", "net.fc.bias"}
    if ocfg.get("head_lr", None) is not None:
        head, body = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            (head if n in head_names else body).append(p)
        params = [
            {"params": body, "lr": ocfg["lr"]},
            {"params": head, "lr": ocfg["head_lr"]},
        ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if ocfg["name"].lower() == "sgd":
        opt = optim.SGD(params, lr=ocfg["lr"], momentum=ocfg.get("momentum", 0.9),
                        weight_decay=ocfg.get("weight_decay", 1e-4),
                        nesterov=ocfg.get("nesterov", True))
    elif ocfg["name"].lower() == "adamw":
        opt = optim.AdamW(params, lr=ocfg["lr"], weight_decay=ocfg.get("weight_decay", 0.05))
    else:
        raise ValueError(f"Unknown optimizer {ocfg['name']}")
    return opt

def build_scheduler(cfg, optimizer, steps_per_epoch):
    scfg = cfg.get("scheduler", {"name":"none"})
    name = scfg.get("name", "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        T = cfg["train"]["epochs"] * steps_per_epoch
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
    if name == "step":
        return optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=scfg.get("milestones", [30,60,80]),
                    gamma=scfg.get("gamma", 0.1))
    raise ValueError(f"Unknown scheduler {name}")

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, criterion, opt, dl, device, scaler, log_interval=50):
    model.train()
    running = 0.0
    seen = 0
    start = time.time()

    pbar = tqdm(enumerate(dl), total=len(dl), desc="train", ncols=100)
    for i, batch in enumerate(dl):
        if isinstance(batch, dict): x, y = batch["jpg"], batch["cls"]
        else:
            if isinstance(batch, list): x, y = batch[0], batch[1]
            else: x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is None:
            loss.backward()
            opt.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        running += loss.item() * y.size(0)
        seen += y.size(0)

        if (i + 1) % log_interval == 0 or (i + 1) == len(dl):
            fps = seen / max(1e-6, (time.time() - start))
            pbar.set_postfix_str(f"loss {running/seen:.4f} | {fps:.0f} samp/s")
    return running / max(1, seen)

@torch.no_grad()
def evaluate(model, criterion, dl, device):
    model.eval()
    total = 0
    loss_sum = 0.0
    correct1 = 0.0
    correct5 = 0.0

    pbar = tqdm(dl, desc="val", ncols=100)
    for batch in dl:
        if isinstance(batch, dict): x, y = batch["jpg"], batch["cls"]
        else:
            if isinstance(batch, list): x, y = batch[0], batch[1]
            else: x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        c1, c5 = topk_correct(logits, y, ks=(1,5))
        correct1 += c1; correct5 += c5
        total += y.size(0)

        pbar.set_postfix_str(f"loss {loss_sum/max(1,total):.4f} | top1 {100*correct1/max(1,total):.2f}% | top5 {100*correct5/max(1,total):.2f}%")
        
    return loss_sum/total, (correct1/total)*100.0, (correct5/total)*100.0

def save_ckpt(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", default="configs/food101.yaml")
    # Handy overrides
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--pretrained", type=str, choices=["true","false"])
    ap.add_argument("--freeze_up_to", type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    # apply simple overrides
    if args.epochs is not None:     cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None: cfg["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:         cfg["optimizer"]["lr"] = args.lr
    if args.pretrained is not None: cfg["model"]["pretrained"] = (args.pretrained.lower()=="true")
    if args.freeze_up_to is not None: cfg["model"]["freeze_up_to"] = args.freeze_up_to

    set_seed(cfg.get("seed"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dl, val_dl, inferred_nc = build_dataloaders(cfg)

    num_classes = cfg["model"].get("num_classes", inferred_nc or 1000)
    model = ResNetBaseline(num_classes=num_classes,
                           pretrained=bool(cfg["model"].get("pretrained", False))).to(device)

    # freezing policy
    maybe_freeze(model, cfg["model"].get("freeze_up_to","none"))

    criterion = build_loss(cfg)
    opt = build_optimizer(cfg, model)

    steps_per_epoch = len(train_dl) if hasattr(train_dl, "__len__") else 1000
    sched = build_scheduler(cfg, opt, steps_per_epoch)

    use_amp = bool(cfg["train"].get("amp", True)) and device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    save_dir = Path(cfg["train"].get("save_dir","ckpts"))
    save_dir.mkdir(parents=True, exist_ok=True)
    # keep a copy of the effective config
    with open(save_dir / "config.effective.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    start_epoch = 0
    best_top1 = -1.0
    resume_path = cfg["train"].get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt: opt.load_state_dict(ckpt["opt"])
        if "scaler" in ckpt and scaler is not None: scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_top1 = ckpt.get("best_top1", best_top1)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    epochs = cfg["train"]["epochs"]
    log_interval = cfg["train"].get("log_interval", 50)
    eval_interval = cfg["train"].get("eval_interval", 1)

    for ep in range(start_epoch, epochs):
        print(f"\nEpoch {ep+1}/{epochs}")
        train_loss = train_one_epoch(model, criterion, opt, train_dl, device, scaler, log_interval)
        if sched is not None:
            # If cosine per-step, step many times; here we step per-epoch unless you switch to per-iter stepping.
            if not isinstance(sched, optim.lr_scheduler.CosineAnnealingLR):
                sched.step()

        if (ep+1) % eval_interval == 0:
            val_loss, top1, top5 = evaluate(model, criterion, val_dl, device)
            print(f"val: loss {val_loss:.4f} | top1 {top1:.2f}% | top5 {top5:.2f}%")

            is_best = top1 > best_top1
            best_top1 = max(best_top1, top1)

            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_top1": best_top1,
                "cfg": cfg,
            }
            save_ckpt(ckpt, save_dir / "last.pt")
            if is_best:
                save_ckpt(ckpt, save_dir / "best.pt")

        # If cosine per-step is desired, you can step sched inside train loop instead.

if __name__ == "__main__":
    main()
