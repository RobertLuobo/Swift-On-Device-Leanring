import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision

from quantizer import Qconv2d, QLinear, Qconv2d_INT, QLinear_INT
from Timer import Timer_logger
# from config import cfg

Timer_logger = Timer_logger()
Timer_logger.log_info("===> training ")


import torch
from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.append("./runs")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset = "CIFAR10"
    dataset = "MNIST"
    batch_size = 256
    test_batch_size = batch_size

    input_size = 28
    epoch = 20
    lr = 0.1
    momentum = 0.9
    seed = 1

    # QAT cfg
    start_QAT_epoch = 19
    num_bits = 8

    # log config
    log_interval = 39
    save_model = False
    no_cuda = False

    # file path cfg
    dataset_root = "../data"
    logger_path = './logger/log'

    # 构建 SummaryWriter
    writer = SummaryWriter(comment='test_your_comment', filename_suffix="_test_your_filename_suffix")


n_count, n_interval = 0, 0
class mnistcnn(nn.Module):
    def __init__(self):
        super(mnistcnn, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=3,stride=1,padding=1 ),
        #     # nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #
        #     # nn.Dropout(),
        #
        #     nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1 ),
        #     # nn.BatchNorm2d(128),
        #     nn.ReLU(inplace=True),
        #
        #     nn.MaxPool2d(stride=2,kernel_size=2)
        # )
        #
        # self.fc = nn.Sequential(
        #     nn.Linear(14*14*128, 1024, bias=False),
        #     nn.ReLU(inplace=True),
        #
        #     # nn.Dropout(),
        #
        #     nn.Linear(1024, 10, bias=True),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1 =  nn.Conv2d(1, 64, kernel_size=3,stride=1,padding=1 )
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3,stride=1,padding=1 )
        self.activaton = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(stride=2,kernel_size=2)
        self.fc1 =  nn.Linear(14*14*128, 1024, bias=False)
        self.fc2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        global n_count, n_interval
        x = self.conv1(x)
        if(n_interval % 39 == 0):  cfg.writer.add_histogram("conv1", x, n_count)

        x = self.activaton(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("relu(conv1)", x, n_count)


        x = self.conv2(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("conv2", x, n_count)
        x = self.activaton(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("relu(conv2)", x, n_count)
        x = self.pool(x)

        x = x.view(-1, 14 * 14 * 128)

        x = self.fc1(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("fc1", x, n_count)
        x = self.activaton(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("relu(fc1)", x, n_count)
        x = self.fc2(x)
        if(n_interval % 39 == 0): cfg.writer.add_histogram("fc2", x, n_count)
        x = self.activaton(x)
        if (n_interval % 39 == 0): cfg.writer.add_histogram("relu(fc2)", x, n_count)

        n_interval += 1
        if(n_interval % 39 == 0): n_count += 1
        return F.log_softmax(x)



def create_model():
    Net = mnistcnn()
    # for param in Net.parameters():
    #     param.requires_grad = False
        # print(param)

    print("cfg/mnistcnn.py")
    from torchsummary import summary

    print(summary(model=Net.to("cuda"), input_size=(1, 28, 28), batch_size=cfg.batch_size,
                  device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    return Net


def convert_model_INT(net):
    net_state_dict = net.state_dict()
    layer_list = []
    TEST_On_else = False
    layer_Sequential = ['conv', 'fc']
    cnt_QLinear = 0
    i = 0
    for outname, modules in net.named_children():
        print(outname)

        if isinstance(modules, nn.Sequential):
            for name, module in modules.named_children():
                print(name, type(module))
                if isinstance(module, nn.Conv2d):
                    if TEST_On_else: print(name, type(module), module.in_channels, module.out_channels)

                    if (False if module.bias is None else True):
                        module = Qconv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                         kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                         dilation=module.dilation, groups=module.groups,
                                         bias=False if module.bias is None else True
                                         )
                    else:
                        module = Qconv2d_INT(in_channels=module.in_channels, out_channels=module.out_channels,
                                         kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                         dilation=module.dilation, groups=module.groups,
                                         bias=False if module.bias is None else True
                                         )

                    getattr(net, layer_Sequential[i])[int(name)] = module
                    print(getattr(net, layer_Sequential[i])[int(name)])
                    layer_list.append(module)

                elif isinstance(module, nn.BatchNorm2d):
                    if TEST_On_else: print(name, type(module), module.num_features)
                    layer_list.append(module)

                elif isinstance(module, nn.Linear):
                    cnt_QLinear += 1
                    if(cnt_QLinear == 2): continue
                    # module = QLinear(in_features=module.in_features, out_features=module.out_features,
                    #                  bias=False if module.bias is None else True)
                    module = QLinear_INT(in_features=module.in_features, out_features=module.out_features,
                                     bias=False if module.bias is None else True)
                    getattr(net, layer_Sequential[i])[int(name)] = module
                    print(getattr(net, layer_Sequential[i])[int(name)])
                    layer_list.append(module)

                else:
                    if TEST_On_else: print(name, type(module))
                    layer_list.append(module)
        else:  # maxpool + fc
            pass
            # if TEST_On:
            # if True:
            #     print("Not Sequential:",modules, type(modules))
            #
            # if isinstance(modules, nn.Linear):
            #     modules = QLinear(in_features=modules.in_features, out_features=modules.out_features,
            #                       bias=False if modules.bias is None else True)
            #
            #     print("print", getattr(net, outname))
            # elif isinstance(modules, nn.Conv2d):
            #     modules = Qconv2d(in_channels=modules.in_channels, out_channels=modules.out_channels,
            #                                         kernel_size=modules.kernel_size, stride=modules.stride,
            #                                         padding=modules.padding,
            #                                         dilation=modules.dilation, groups=modules.groups,
            #                                         bias=False if modules.bias is None else True
            #                                         )
            # else:
            #     # raise RuntimeError('Not Linear layer')
            #     pass
            # layer_list.append(modules)

        i += 1
    net.load_state_dict(net_state_dict)
    print(len(layer_list))
    # print(layer_list)
    return net

def convert_model(net):
    net_state_dict = net.state_dict()
    layer_list = []
    TEST_On_else = False
    layer_Sequential = ['conv', 'fc']
    i = 0
    for outname, modules in net.named_children():
        print(outname)

        if isinstance(modules, nn.Sequential):
            for name, module in modules.named_children():
                print(name, type(module))
                if isinstance(module, nn.Conv2d):
                    if TEST_On_else: print(name, type(module), module.in_channels, module.out_channels)

                    module = Qconv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                     kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                     dilation=module.dilation, groups=module.groups,
                                     bias=False if module.bias is None else True
                                     )

                    getattr(net, layer_Sequential[i])[int(name)] = module
                    print(getattr(net, layer_Sequential[i])[int(name)])
                    layer_list.append(module)

                elif isinstance(module, nn.BatchNorm2d):
                    if TEST_On_else: print(name, type(module), module.num_features)
                    layer_list.append(module)

                elif isinstance(module, nn.Linear):
                    module = QLinear(in_features=module.in_features, out_features=module.out_features,
                                     bias=False if module.bias is None else True)

                    getattr(net, layer_Sequential[i])[int(name)] = module
                    print(getattr(net, layer_Sequential[i])[int(name)])
                    layer_list.append(module)

                else:
                    if TEST_On_else: print(name, type(module))
                    layer_list.append(module)
        else:  # maxpool + fc
            # if TEST_On:
            if True:
                print("Not Sequential:",modules, type(modules))

            if isinstance(modules, nn.Linear):
                modules = QLinear(in_features=modules.in_features, out_features=modules.out_features,
                                  bias=False if modules.bias is None else True)

                print("print", getattr(net, outname))
            elif isinstance(modules, nn.Conv2d):
                modules = Qconv2d(in_channels=modules.in_channels, out_channels=modules.out_channels,
                                                    kernel_size=modules.kernel_size, stride=modules.stride,
                                                    padding=modules.padding,
                                                    dilation=modules.dilation, groups=modules.groups,
                                                    bias=False if modules.bias is None else True
                                                    )
            else:
                # raise RuntimeError('Not Linear layer')
                pass
            layer_list.append(modules)



        i += 1
    net.load_state_dict(net_state_dict)
    print(len(layer_list))
    # print(layer_list)
    return net

def freeze(net):
    for param in net.parameters():
        param.requires_grad = False

    # for param in net.fc2.parameters():
    #     param.requires_grad = True



class Trainer:
    def __init__(self):
        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.test_batch_size = cfg.test_batch_size

        self.epochs = cfg.epoch
        self.lr = cfg.lr
        self.momentum = cfg.momentum
        self.input_size = cfg.input_size

        self.log_interval = cfg.log_interval
        self.save_model = cfg.save_model

        self.Net = create_model()
        self.QAT_flag = True

        self.iter_count = 0

        if cfg.dataset == "CIFAR10":
            transform = transforms.Compose(
                [
                    transforms.Resize(cfg.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ]
            )

            trainset = datasets.CIFAR10(root=cfg.dataset_root, train=True,
                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                                       shuffle=True, num_workers=0)

            testset = datasets.CIFAR10(root=cfg.dataset_root, train=False,
                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size,
                                                      shuffle=False, num_workers=0)
        elif cfg.dataset == "MNIST":
            transform = transforms.Compose(
                [
                    # transforms.Resize(cfg.input_size),
                    transforms.ToTensor()
                ]
            )

            trainset = datasets.FashionMNIST(root=cfg.dataset_root, train=True,
                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                                       shuffle=True, num_workers=0)

            testset = datasets.FashionMNIST(root=cfg.dataset_root, train=False,
                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size,
                                                      shuffle=False, num_workers=0)

    def adjust_learning_rate(self, optimizer, epoch):
        """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
        adjust_list = [4, 10]
        if epoch in adjust_list:
            for param_group in optimizer.param_groups:
                print("adjust_learning_rate function:", epoch, " learning rate:", param_group['lr'])
                param_group['lr'] = param_group['lr'] * 0.1

    @staticmethod
    def visualise(x, axs):
        x = x.view(-1).cpu().numpy()
        axs.hist(x)

    def print_requires_grad(self):
        for param in self.Net.parameters():
            if param.requires_grad == True:
                print(param.size(), param)

    def train(self, args, model, optimizer, epoch):
        correct = 0
        model.train()
        Timer_logger.start()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            self.iter_count += 1
            # pred = output.argmax(dim=1, keepdim=True)
            # correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % args["log_interval"] == 0:
                Timer_logger.log()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()
                ))
                self.evaluate(args, model)
                Timer_logger.start()
            # 记录数据，保存于event file
            cfg.writer.add_scalars("Loss", {"Train": loss.item()}, self.iter_count)


    def evaluate(self, args, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        cfg.writer.add_scalars("Accuracy", {"Train":  correct/10000 }, self.iter_count)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))


    def run(self):
        Timer_logger.start()

        seed = cfg.seed
        no_cuda = cfg.no_cuda

        use_cuda = not no_cuda and torch.cuda.is_available()

        torch.manual_seed(seed)

        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        self.Net.to( self.device)
        optimizer = optim.SGD(self.Net.parameters(), lr= self.lr, momentum= self.momentum)
        args = {}
        args["log_interval"] =  self.log_interval

        for epoch in range(1,  self.epochs + 1):
            # print(epoch, cfg.start_QAT_epoch, epoch >= cfg.start_QAT_epoch, self.QAT_flag)

            Timer_logger.log_info("normal:")
            Timer_logger.start()
            self.train(args, self.Net,  optimizer, epoch)
            Timer_logger.log()

            # Timer_logger.log_info("test")
            # Timer_logger.start()
            # self.evaluate(args, self.Net)
            # Timer_logger.log()

        if(self.save_model):
            torch.save(self.Net.state_dict(), "mnist_cnn.pt")

        Timer_logger.log()
        # return model

    def run_INT(self):
        Timer_logger.start()

        seed = cfg.seed
        no_cuda = cfg.no_cuda

        use_cuda = not no_cuda and torch.cuda.is_available()

        torch.manual_seed(seed)

        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        self.Net.to( self.device)
        optimizer = optim.SGD(self.Net.parameters(), lr= self.lr, momentum= self.momentum)
        args = {}
        args["log_interval"] =  self.log_interval

        for epoch in range(1,  self.epochs + 1):

            if(epoch > cfg.start_QAT_epoch and self.QAT_flag):
                break
                self.Net = convert_model_INT(self.Net).cpu()
                # freeze(Trainer.Net)
                self.device = "cpu"
                print(self.Net)
                self.Net.to(self.device)
                self.QAT_flag = False

                optimizer = optim.SGD(self.Net.parameters(), lr=self.lr, momentum=self.momentum)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.01

            print(epoch, cfg.start_QAT_epoch, self.QAT_flag, self.device,
                  (epoch >= cfg.start_QAT_epoch and self.QAT_flag)
                  )

                # for k, param in self.Net.parameters():
                #     if param.requires_grad == True:
                #         print(k, param)

            Timer_logger.log_info("normal:")
            Timer_logger.start()
            self.train(args, self.Net,  optimizer, epoch)
            Timer_logger.log()

            self.adjust_learning_rate(optimizer, epoch)
            # Timer_logger.log_info("test")
            # Timer_logger.start()
            # self.evaluate(args, self.Net)
            # Timer_logger.log()

        if(self.save_model):
            torch.save(self.Net.state_dict(), "mnist_cnn.pt")

        Timer_logger.log()
        # return model

Trainer = Trainer()
print(Trainer.Net)
# convert_model(Trainer.Net)
# convert_model_INT(Trainer.Net)

'''''
1. 就是找个simple，shallow的network，看一下是不是就能int 8直接work了、以及
2. 量化前后中间结果的gap'''
# freeze(Trainer.Net)
print(Trainer.Net)

Trainer.run_INT()








