import torch

class cfg:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "CIFAR10"
    # dataset = "MNIST"
    batch_size = 128
    test_batch_size = batch_size

    input_size = 224
    epoch = 20
    lr = 0.1
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


