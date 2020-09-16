import torch
from torch.utils.tensorboard import SummaryWriter
# import sys
# sys.path.append("./runs")

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # dataset = "CIFAR10"
    dataset = "MNIST"
    batch_size = 256
    test_batch_size = batch_size

    input_size = 28
    epoch = 20
    lr = 0.01
    momentum = 0.9
    seed = 1

    # QAT cfg
    start_QAT_epoch = 0
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





