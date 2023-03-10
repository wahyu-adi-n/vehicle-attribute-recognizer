from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from datasets.base import DataModuleBase


class CarsDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_bs = cfg['train']['batch_size']
        self.val_bs = cfg['val']['batch_size']
        self.test_bs = cfg['test']['batch_size']

        self.train_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=cfg['dataset']['augmentation']['rotation_range']
            ),
            transforms.RandomAdjustSharpness(
                sharpness_factor=cfg['dataset']['augmentation']['sharpness_factor'],
                p=0.5),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomPerspective(
                distortion_scale=cfg['dataset']['augmentation']['distortion_scale'], p=0.5),
            transforms.RandomPosterize(
                bits=cfg['dataset']['augmentation']['bits'], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])

        self.test_val_transforms = transforms.Compose([
            transforms.Resize(cfg['model']['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['std'])
        ])
        self.prepare_dataset()

    def prepare_dataset(self):
        self.train_set = ImageFolder(root=self.cfg['dataset']['train_dir'],
                                     transform=self.train_transforms)
        self.val_set = ImageFolder(root=self.cfg['dataset']['val_dir'],
                                   transform=self.test_val_transforms)
        self.test_set = ImageFolder(root=self.cfg['dataset']['test_dir'],
                                    transform=self.test_val_transforms)
        # return self.train_set, self.val_set, self.test_set

    def train_dataloader(self):
        kwargs = dict(
            batch_size=self.train_bs,
            shuffle=True,
            num_workers=self.cfg['dataset']['num_workers']
        )
        self.train_dl = DataLoader(self.train_set, pin_memory=True, **kwargs)
        return self.train_dl

    def val_dataloader(self):
        kwargs = dict(
            batch_size=self.val_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers']
        )
        self.val_dl = DataLoader(self.val_set,  pin_memory=True, **kwargs)
        return self.val_dl

    def test_dataloader(self):
        kwargs = dict(
            batch_size=self.test_bs,
            shuffle=False,
            num_workers=self.cfg['dataset']['num_workers']
        )
        self.test_dl = DataLoader(self.test_set,  pin_memory=True, **kwargs)
        return self.test_dl

    def get_classes(self):
        return self.train_set.classes
