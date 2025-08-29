from torchvision.datasets import ImageNet
from torchvision import transforms

def make_imagenet_fs(root, split, img_size=224):
    aug = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ]) if split=="val" else transforms.Compose([
        transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])
    return ImageNet(root=root, split=split, transform=aug)
