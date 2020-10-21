import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models


from quantizer import  Qconv2d_INT, QLinear_INT
from Timer import Timer_logger
from config import cfg
from dataset.mnist import load_mnist
Timer_logger = Timer_logger()
Timer_logger.log_info("===> training ")
import time
print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))


class mnistcnn(nn.Module):
    def __init__(self):
        super(mnistcnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3,stride=1,padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Dropout(),

            nn.MaxPool2d(stride=2, kernel_size=2),

            # nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            # nn.Linear(10*10*256, 1024, bias=True),
            nn.Linear(12 * 12 * 64, 1024, bias=True),
            nn.ReLU(inplace=True),

            # nn.Dropout(),

            nn.Linear(1024, 10, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        # x = x.view(-1, 10 * 10 * 256)
        x = x.view(-1, 12 * 12 * 64)
        x = self.fc(x)
        return F.log_softmax(x)


def create_model():
    Net = mnistcnn()
    # for param in Net.parameters():
    #     param.requires_grad = False
        # print(param)

    # print("cfg/mnistcnn.py")
    from torchsummary import summary

    print(summary(model=Net.to("cpu"), input_size=(1, 28, 28), batch_size=cfg.batch_size,
                  device="cpu"))  # model, input_size(channel, H, W), batch_size, device
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
                    '''Paper Trick: Last fc layer do not quantization will get better accuracy'''
                    # if(cnt_QLinear == 2): continue


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


        i += 1
    net.load_state_dict(net_state_dict)
    print(len(layer_list))
    # print(layer_list)
    return net


def freeze(net):
    for param in net.parameters():
        param.requires_grad = False


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
        self.optimizer = optim.SGD(self.Net.parameters(), lr=self.lr, momentum=self.momentum)
        print("SGD\n")
        # self.optimizer = optim.Adam(self.Net.parameters(), lr=self.lr)
        # print("Adam\n")
        # self.scheduler_lr = optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)

        self.milestones = [5+1, 10+1]
        # self.milestones = [10]
        self.scheduler_lr =optim.lr_scheduler.MultiStepLR(self.optimizer , milestones=self.milestones , gamma=0.1)
        self.QAT_flag = True

        self.iter_count = 0

        self.Train_Acc_list = []
        self.Test_Acc_list = []
        self.Train_Loss_list = []
        self.Test_Loss_list = []

        if cfg.dataset == "MNIST":
            transform = transforms.Compose(
                [
                    # transforms.Resize(cfg.input_size),
                    transforms.ToTensor()
                ]
            )

            trainset = torchvision.datasets.MNIST(root=cfg.dataset_root, train=True,
                                        download=True, transform=transform)
            self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size,
                                                       shuffle=True, num_workers=0)

            testset = torchvision.datasets.MNIST(root=cfg.dataset_root, train=False,
                                       download=True, transform=transform)
            self.test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size,
                                                      shuffle=False, num_workers=0)


    def print_requires_grad(self):
        for param in self.Net.parameters():
            if param.requires_grad == True:
                print(param.size(), param)


    def train(self, args, model, optimizer, epoch):
        train_loss = 0
        model.train()
        # Timer_logger.start()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            output = model(data)
            loss = F.cross_entropy(output, target)

            '''if loss change to Nan-value, the whole training crash'''
            if(torch.any(torch.isnan(loss))): print((loss.size()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True)
            train_loss += loss
            train_correct = pred.eq(target.view_as(pred)).sum().item()
            self.iter_count += 1

            if batch_idx % args["log_interval"] == 0:
                # Timer_logger.log()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain_Loss: {:.6f}\tTrain_Acc[{}/{}]: {:.3f}%'.
                    format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item(),
                        train_correct, self.batch_size, 100 * train_correct / self.batch_size
                    )
                )
                self.Train_Acc_list.append(100 * train_correct / self.batch_size)
                self.Train_Loss_list.append(loss.item())
                self.vaildation(args, model)
                # Timer_logger.start()
            # 记录数据，保存于event file
            cfg.writer.add_scalars("Loss", {"Train": loss.item()}, self.iter_count)


    def vaildation(self, args, model):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        cfg.writer.add_scalars("Accuracy", {"Train":  correct/10000 }, self.iter_count)

        print('\nTest set: Average vaildation loss: {:.4f}, vaild_Acc: {}/{} ({}%)\n'.
            format(
                test_loss, correct, len(self.test_loader.dataset),
                100. * correct / len(self.test_loader.dataset)
            )
        )

        self.Test_Acc_list.append(100. * correct / len(self.test_loader.dataset))
        self.Test_Loss_list.append(test_loss)
        model.train()

    def print_for_draw(self):
        print("Train_Acc_list: ", self.Train_Acc_list)
        print("Train_Loss_list:", self.Train_Loss_list)
        print("Test_Acc_list: ", self.Test_Acc_list)
        print("Test_Loss_list:", self.Test_Loss_list)


    def run_INT(self):
        seed = cfg.seed
        no_cuda = cfg.no_cuda
        use_cuda = not no_cuda and torch.cuda.is_available()
        # torch.manual_seed(seed)

        kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

        self.Net.to( self.device)

        args = {}
        args["log_interval"] =  self.log_interval

        self.print_for_draw()
        for epoch in range(1,  self.epochs + 1):

            if(epoch > cfg.start_QAT_epoch and self.QAT_flag):
                self.Net = convert_model_INT(self.Net).cpu()

                self.device = "cpu"
                print(self.Net)
                self.Net.to(self.device)
                self.QAT_flag = False
                self.optimizer = optim.SGD(self.Net.parameters(),
                                           lr= self.optimizer.state_dict()['param_groups'][0]['lr'] ,
                                           momentum=self.optimizer.state_dict()['param_groups'][0]['momentum'])

                # self.optimizer = optim.SGD(self.Net.parameters(), lr=self.lr, momentum=self.momentum)
                # for param_group in self.optimizer.param_groups:
                #     param_group['lr'] = param_group['lr'] * 0.01
                # print("if(epoch > cfg.start_QAT_epoch and self.QAT_flag) ",
                #       self.optimizer.state_dict()['param_groups'][0]['lr'])

            self.scheduler_lr.step()
            print("Before epoch training:",
                  self.optimizer.state_dict()['param_groups'][0]['lr'])
            print(
                epoch, cfg.start_QAT_epoch, self.QAT_flag, self.device,
                (epoch >= cfg.start_QAT_epoch and self.QAT_flag)
                )


            print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.train(args, self.Net, self.optimizer, epoch)
            # self.vaildation(args,self.Net)
        self.print_for_draw()

        # return model

Trainer = Trainer()

'''Fake QAT'''
# convert_model(Trainer.Net)
# print(Trainer.Net)
# Trainer.run()



'''INT8 * INT8'''
print(Trainer.Net)
# Trainer.print_requires_grad()
Trainer.run_INT()








