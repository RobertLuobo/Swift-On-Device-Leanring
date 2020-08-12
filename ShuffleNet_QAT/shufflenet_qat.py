import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision

from quantizer import Qconv2d, QLinear
from Timer import Timer_logger
from config import cfg
import shufflenet
Timer_logger = Timer_logger()
Timer_logger.log_info("===> training ")

def create_model():
    # Net = torchvision.models.shufflenet_v2_x1_0(pretrained=True, num_classes=1000)
    Net = shufflenet.shufflenet_v2_x1_0(pretrained=True, num_classes=1000)
    for param in Net.parameters():
        param.requires_grad = False
        # print(param)
    fc_features = Net.fc.in_features
    Net.fc = nn.Sequential(
                nn.Linear(fc_features, 10),
            )
    for param in Net.fc.parameters():
        param.requires_grad = True
    # Net.fc = nn.Linear(fc_features, 10)

    print("cfg/shufflenetv2.py")
    # print()
    # from torchsummary import summary

    # print(summary(model=Net.to("cuda"), input_size=(3, 224, 224), batch_size=128,
    #               device="cuda"))  # model, input_size(channel, H, W), batch_size, device
    return Net

def convert_model(net):
    net_state_dict = net.state_dict()
    layer_list = []
    layer_Sequential = ['conv1', 'maxpool', 'stage2', 'stage3', 'stage4', 'conv5', 'fc']

    # flag for print info
    TEST_On = False
    TEST_On_else = False


    i = 0

    for outname, modules in net.named_children():
        print(outname)

        if isinstance(modules, nn.Sequential):
            for name, module in modules.named_children():
                print(name, type(module))
                # print("name:",name, type(name))
                # if isinstance(module, models.shufflenetv2.InvertedResidual):
                if isinstance(module, shufflenet.InvertedResidual):#stage2 + stage3 + stage4
                    for key, layers in module.named_children():
                        # print("key:",key, type(key))
                        if isinstance(layers, nn.Sequential):
                            for k, layer in layers.named_children():
                                # print("k:",k, type(k))
                                if isinstance(layer, nn.Conv2d):
                                    if TEST_On:
                                        print(k, type(layer), layer.in_channels, layer.out_channels)
                                        print(layer.weight)

                                    layer = Qconv2d(in_channels=layer.in_channels, out_channels=layer.out_channels,
                                                    kernel_size=layer.kernel_size, stride=layer.stride,
                                                    padding=layer.padding,
                                                    dilation=layer.dilation, groups=layer.groups,
                                                    bias=False if layer.bias is None else True
                                                    )
                                    getattr((getattr(net, layer_Sequential[i])[int(name)]), str(key))[int(k)] = layer
                                    print(getattr((getattr(net, layer_Sequential[i])[int(name)]), str(key))[int(k)])
                                    layer_list.append(layer)

                                elif isinstance(layer, nn.BatchNorm2d):
                                    if TEST_On: print(k, type(layer), layer.num_features)
                                    layer_list.append(layer)

                                elif isinstance(layer, nn.Linear):
                                    raise RuntimeError('Linear layer')

                                else:
                                    if TEST_On: print(k, type(layer))
                                    layer_list.append(layer)

                else:# conv1 + conv5
                    # print("else", name, type(module))
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
                print("is Not Sequential", modules, type(modules))

            if isinstance(modules, nn.Linear):
                modules = QLinear(in_features=modules.in_features, out_features=modules.out_features,
                                  bias=False if modules.bias is None else True)

                print(type(getattr(net, layer_Sequential[i])))
                # op = operator.attrgetter(layer_Sequential[i])
                # print(op(net) ,type(op(net)))
                # op(net) = modules
                # getattr(net, layer_Sequential[i])= modules
                print(getattr(net, layer_Sequential[i]))

                raise RuntimeError('Linear layer')
            elif isinstance(modules, nn.MaxPool2d):
                pass
            else:
                raise RuntimeError('Not Linear layer')
            layer_list.append(modules)


        i += 1
    net.load_state_dict(net_state_dict)
    print(len(layer_list))
    # print(layer_list)
    return net

def freeze(net):
    for param in net.parameters():
        param.requires_grad = False

    for param in net.fc.parameters():
        param.requires_grad = True




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
                    transforms.Resize(cfg.input_size),
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
    @staticmethod
    def visualise(x, axs):
        x = x.view(-1).cpu().numpy()
        axs.hist(x)

    def print_requires_grad(self):
        for param in self.Net.parameters():
            if param.requires_grad == True:
                print(param.size(), param)

    def train(self, args, model, optimizer, epoch):
        model.train()
        Timer_logger.start()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % args["log_interval"] == 0:
                Timer_logger.log()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()
                ))
                self.evaluate(args, model)
                Timer_logger.start()


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
            # if(epoch >= cfg.start_QAT_epoch and self.QAT_flag):
            #     self.Net = convert_model(self.Net)
            #     freeze(Trainer.Net)
            #     print(self.Net)
            #     self.Net.to(self.device)
            #     self.QAT_flag = False

                # for k, param in self.Net.parameters():
                #     if param.requires_grad == True:
                #         print(k, param)

            Timer_logger.log_info("normal:")
            Timer_logger.start()
            self.train(args, self.Net,  optimizer, epoch)
            Timer_logger.log()

            # Timer_logger.log_info("test")
            # Timer_logger.start()
            # self.evaluate(args, self.Net)
            # Timer_logger.log()

        if ( self.save_model):
            torch.save(self.Net.state_dict(), "mnist_cnn.pt")

        Timer_logger.log()
        # return model


Trainer = Trainer()

print(Trainer.Net)
convert_model(Trainer.Net)
freeze(Trainer.Net)
print(Trainer.Net)


Trainer.run()


