import torch
import torchvision

# print(torch.__version__)
# print(torchvision.__version__)


from torchvision import datasets, transforms, models
from config import cfg
# import torch.utils.data.dataloader as Data
transform = transforms.Compose(
        [
            # transforms.Resize(cfg.input_size),
            transforms.ToTensor()
        ]
    )

# trainset = torchvision.datasets.MNIST(root=cfg.dataset_root, train=True,
#                             download=True, transform=torchvision.transforms.ToTensor())
# train_loader =  Data.DataLoader(trainset, batch_size=cfg.batch_size,
#                                            shuffle=True, num_workers=0)
#
# testset =torchvision.datasets.MNIST(root=cfg.dataset_root, train=False,
#                            download=True, transform=torchvision.transforms.ToTensor())
# test_loader =  Data.DataLoader(testset, batch_size=cfg.test_batch_size,
#                                           shuffle=False, num_workers=0)

trainset = torchvision.datasets.MNIST(root=cfg.dataset_root, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                           shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root=cfg.dataset_root, train=False,
                           download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size,
                                          shuffle=False, num_workers=0)
torch.set_printoptions(profile="full")
for  (data, target) in (train_loader):
    data, target = data.to("cpu"), target.to("cpu")
    # print(data.size(), data)\
    test_data = torch.zeros_like(data, device="cpu")
    print(torch.eq(data, test_data))




    # print(test_data.size(), data.size(), test_data,)