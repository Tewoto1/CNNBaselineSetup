import webdataset as wds
from torchvision import transforms
from torch.utils.data import DataLoader

def make_imagenet_wds(url_pattern, split, img_size=224, batch_size=256, num_workers=8):
    aug_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    aug_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    aug = aug_val if split=="val" else aug_train

    # e.g., url_pattern = "https://my-bucket/imagenet-train-{000000..001023}.tar"
    dataset = (wds.WebDataset(url_pattern, shardshuffle=(split=="train"))
                 .decode("pil")
                 .to_tuple("jpg;jpeg;png", "cls")  # keys inside each sample
                 .map_tuple(aug, lambda x: x))

    dl = DataLoader(dataset.batched(batch_size),
                    num_workers=num_workers, pin_memory=True)
    return dl
