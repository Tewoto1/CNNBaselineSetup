import os, argparse, torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchmetrics.functional as F  # optional

from models.resnet_baseline import ResNetBaseline
from losses.cross_entropy import LabelSmoothingCE
from datasets.imagenet_fs import make_imagenet_fs
from datasets.imagenet_wds import make_imagenet_wds

def get_loaders(args):
    if args.wds:  # streaming
        train = make_imagenet_wds(args.train_urls, "train", args.img_size, args.batch_size, args.workers)
        val   = make_imagenet_wds(args.val_urls,   "val",   args.img_size, args.batch_size, args.workers)
        return train, val
    # filesystem
    train_ds = make_imagenet_fs(args.data_root, "train", args.img_size)
    val_ds   = make_imagenet_fs(args.data_root, "val", args.img_size)
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    val   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.workers, pin_memory=True)
    return train, val

def train_one_epoch(model, criterion, opt, dl, device, scaler):
    model.train()
    running = 0.0
    for batch in tqdm(dl, desc="train"):
        if isinstance(batch, dict):  # TorchData style
            x, y = batch["jpg"], batch["cls"]
        else:
            if isinstance(batch, list): x, y = batch[0], batch[1]
            else: x, y = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).board = None
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        running += loss.item()
    return running / max(1,len(dl))

@torch.no_grad()
def evaluate(model, criterion, dl, device):
    model.eval()
    top1 = top5 = 0; n=0; val_loss=0.0
    for batch in tqdm(dl, desc="val"):
        if isinstance(batch, dict): x, y = batch["jpg"], batch["cls"]
        else:
            if isinstance(batch, list): x, y = batch[0], batch[1]
            else: x, y = batch
        x, y = x.to(device), y.to(device)
        logits = model(x)
        val_loss += criterion(logits, y).item()
        preds = torch.softmax(logits, dim=-1)
        top1 += (preds.argmax(1) == y).sum().item()
        top5 += torch.topk(preds, 5, dim=1).indices.eq(y.unsqueeze(1)).any(1).sum().item()
        n += y.size(0)
    return val_loss/max(1,len(dl)), top1/n*100, top5/n*100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/imagenet")
    ap.add_argument("--wds", action="store_true")
    ap.add_argument("--train_urls", type=str, default="")
    ap.add_argument("--val_urls", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    train_dl, val_dl = get_loaders(args)
    model = ResNetBaseline(num_classes=1000, pretrained=args.pretrained).to(device)
    criterion = LabelSmoothingCE(0.1)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(args.epochs):
        tl = train_one_epoch(model, criterion, opt, train_dl, device, scaler)
        vl, t1, t5 = evaluate(model, criterion, val_dl, device)
        print(f"Epoch {ep:03d} | train {tl:.3f} | val {vl:.3f} | top1 {t1:.2f}% | top5 {t5:.2f}%")
        torch.save({"ep": ep, "model": model.state_dict()}, f"ckpts/ep_{ep:03d}.pt")

if __name__ == "__main__": main()
