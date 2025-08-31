from torchvision.datasets import Food101
from torchvision import transforms

def make_food101_fs(root, split, img_size=224):
    # Food-101 has explicit train/test splits internally
    train = (split == "train")
    aug = transforms.Compose([
        transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ]) if train else transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    return Food101(root=root, split="train" if train else "test",
                   transform=aug, download=True)
