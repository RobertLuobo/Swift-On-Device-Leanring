import torch
import torch.nn.functional as F
from collections import namedtuple


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

def visualise(x, axs):
    x = x.view(-1).cpu().numpy()
    axs.hist(x)

def calcScaleZeroPoint(min_val, max_val, num_bits=8):
    # Calc Scale and zero point of next
    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def calcScaleZeroPointSym(min_val, max_val, num_bits=8):
    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    return scale, 0

from config import cfg
n_count = 0

''''
Func: qunatier 
'''
def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    global n_count
    cfg.writer.add_histogram("input", x, n_count)
    cfg.writer.add_scalars("input_scale", {"input_scale": scale}, n_count)
    cfg.writer.add_scalars("input_zero_point", {"input_zero_point": zero_point}, n_count)
    n_count += 1

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

input_n_count = 0
''''
Func: qunatier for draw the input date in tensorboard
'''
def quantize_tensor_input(x, num_bits=8, min_val=None, max_val=None):

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    global input_n_count
    cfg.writer.add_histogram("input", x, n_count)
    cfg.writer.add_scalars("input_scale", {"input_scale": scale}, n_count)
    cfg.writer.add_scalars("input_zero_point", {"input_zero_point": zero_point}, n_count)
    input_n_count += 1

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

weight_n_count = 0

''''
Func: qunatier for draw the weight date in tensorboard
'''
def quantize_tensor_weight(x, num_bits=8, min_val=None, max_val=None):

    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2. ** num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()

    global weight_n_count
    cfg.writer.add_histogram("weight", x, n_count)
    cfg.writer.add_scalars("weight_scale", {"weight_scale": scale}, n_count)
    cfg.writer.add_scalars("weight_zero_point", {"weight_zero_point": zero_point}, n_count)
    weight_n_count += 1

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


def quantize_tensor_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    max_val = max(abs(min_val), abs(max_val))
    qmin = 0.
    qmax = 2. ** (num_bits - 1) - 1.

    scale = max_val / qmax

    q_x = x / scale

    q_x.clamp_(-qmax, qmax).round_()
    q_x = q_x.round()
    return QTensor(tensor=q_x, scale=scale, zero_point=0)


def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())



class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output,  num_bits=8, min_val=None, max_val=None):
        # x = grad_output
        # x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        # x = dequantize_tensor(x)
        # grad_output = x
        # print("backward")
        return grad_output

'''
Fake quantization conv2d
'''
class Qconv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)

    def conv2d_forward(self, input, num_bits=8, min_val=None, max_val=None):
        output = torch.nn.functional.conv2d(FakeQuantOp.apply(input), FakeQuantOp.apply(self.weight), self.bias, self.stride, self.padding,
                                          self.dilation, self.groups)
        # print(output)

        return output

    def forward(self, input):
        return self.conv2d_forward(input)

'''
Fake quantization Linear
'''
class QLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)

    def _linear_forward(self, input, num_bits=8, min_val=None, max_val=None):
        output = torch.nn.functional.linear(FakeQuantOp.apply(input), FakeQuantOp.apply(self.weight), self.bias)
        return output

    def forward(self, input):
        return self._linear_forward(input)

Conv_count = 0
fc_count = 0

'''
Dong Xin paper uniform quantizer method: alpha * {0 , 1/127, -1/127, .... +1, -1 } by APoT
formula:   
\mathcal{Q}^{u}(\alpha, b)=\alpha \times\left\{0, \frac{\pm 1}{2^{b-1}-1}, \frac{\pm 2}{2^{b-1}-1}, \frac{\pm 3}{2^{b-1}-1}, \ldots,\pm 1\right\}
'''
class Qconv2d_INT(torch.nn.Conv2d):
    # @staticmethod
    # def convert(other):
    #     if not isinstance(other, torch.nn.Conv2d):
    #         raise TypeError("Expected a torch.nn.Conv2d ! Receive:  {}".format(other.__class__))
    #     return Qconv2d(other.in_channels, other.out_channels, other.kernel_size, stride=other.stride,
    #                      padding=other.padding, dilation=other.dilation, groups=other.groups,
    #                      bias=other.bias)


    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)


    def conv2d_forward(self, input, num_bits=8, min_val=None, max_val=None):

        # step1 : discrete
        q_input = quantize_tensor_input(input, num_bits=num_bits,
                                  min_val=torch.min(input), max_val=torch.max(input))

        q_weight = quantize_tensor_weight(self.weight, num_bits=num_bits,
                                   min_val=torch.min(self.weight), max_val=torch.max(self.weight))


        output = torch.nn.functional.conv2d(q_input.tensor.type(torch.IntTensor),
                                            q_weight.tensor.type(torch.IntTensor),
                                            self.bias, self.stride, self.padding, self.dilation, self.groups)

        # step2 : find out alpha
        input_alpha = (input.max() - input.min()) / 2
        weight_alpha = (self.weight.max() - self.weight.min()) / 2

        global Conv_count
        Conv_count += 1
        cfg.writer.add_scalars("Conv_input_alpha", {"Conv_input_alpha": input_alpha}, Conv_count)
        cfg.writer.add_scalars("Conv_weight_alpha", {"Conv_weight_alpha": input_alpha}, Conv_count)

        # Step3 : calibration, add decimal point
        decimal_point = 2 ** num_bits - 1
        output = output / (decimal_point ** 2) * input_alpha * weight_alpha

        # Step4 :  dequantizer
        Qoutput = QTensor(tensor=output, scale=q_weight.scale,zero_point=(q_weight.zero_point))
        output = dequantize_tensor(Qoutput)


        # Qoutput = QTensor(tensor=output, scale=q_input.scale, zero_point=q_input.zero_point)
        # Qoutput = QTensor(tensor=output, scale=q_weight.scale ,
        #                   zero_point=(q_weight.zero_point+q_input.zero_point)/2)
        #
        # output = dequantize_tensor(Qoutput)

        # print("q_input scale:", q_input.scale, "q_weigth scale:", q_weight.scale)
        # output = torch.nn.functional.conv2d(FakeQuantOp.apply(input), FakeQuantOp.apply(self.weight), self.bias, self.stride, self.padding,
        #                                   self.dilation, self.groups)
        # print(output)

        return output

    def forward(self, input):
        return self.conv2d_forward(input)


'''
Dong Xin paper uniform quantizer method: alpha * {0 , 1/127, -1/127, .... +1, -1 } by APoT
'''
class QLinear_INT(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)

    def _linear_forward(self, input, num_bits=8, min_val=None, max_val=None):
        # step1 : discrete
        q_input = quantize_tensor_input(input, num_bits=num_bits,
                                  min_val=torch.min(input), max_val=torch.max(input))
        # q_input = dequantize_tensor(q_input)
        q_weight = quantize_tensor_weight(self.weight, num_bits=num_bits,
                                   min_val=torch.min(self.weight), max_val=torch.max(self.weight))

        output = torch.nn.functional.linear(q_input.tensor.type(torch.IntTensor),
                                            q_weight.tensor.type(torch.IntTensor),
                                            self.bias)

        # Qoutput = QTensor(tensor=output, scale=q_input.scale, zero_point=q_input.zero_point)
        # Qoutput = QTensor(tensor=output, scale=q_weight.scale * q_input.scale,
        #                   zero_point=(q_weight.zero_point + q_input.zero_point) / 2)
        #
        # output = dequantize_tensor(Qoutput)

        # step2 : find out alpha
        input_alpha = (input.max() - input.min()) / 2
        weight_alpha = (self.weight.max() - self.weight.min()) / 2

        global fc_count
        fc_count += 1
        cfg.writer.add_scalars("fc_input_alpha", {"fc_input_alpha": input_alpha}, fc_count)
        cfg.writer.add_scalars("fc_weight_alpha", {"fc_weight_alpha": input_alpha}, fc_count)

        # Step3 : calibration, add decimal point
        decimal_point = 2 ** num_bits - 1
        output = output / (decimal_point ** 2) * input_alpha * weight_alpha

        # Step4 :  dequantizer
        Qoutput = QTensor(tensor=output, scale=q_weight.scale,zero_point=(q_weight.zero_point))
        output = dequantize_tensor(Qoutput)

        return output


    def forward(self, input):
        return self._linear_forward(input)

''''
Func: this Conv2d_Normal same as Conv2d, for tensorboard    
'''
class Conv2d_Normal(torch.nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                 padding=padding, dilation=dilation, groups=groups, bias=bias)

    def conv2d_forward(self, input, num_bits=8, min_val=None, max_val=None):

        output = torch.nn.functional.conv2d(input, self.weight,
                                            self.bias, self.stride, self.padding, self.dilation, self.groups)

        global Conv_count
        Conv_count += 1

        cfg.writer.add_scalars("Conv2d_Normal_input", {"Conv2d_Normal_input": input}, Conv_count)

        return output

    def forward(self, input):
        return self.conv2d_forward(input)


''''
Func: this Linear_Normal same as Linear, for tensorboard 
'''
class Linear_Normal(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        torch.nn.Linear.__init__(self, in_features, out_features, bias=bias)

    def _linear_forward(self, input, num_bits=8, min_val=None, max_val=None):

        output = torch.nn.functional.linear(input, self.weight, self.bias)
        global fc_count
        fc_count+= 1
        cfg.writer.add_scalars("Linear_Normal_input", {"Linear_Normal_input": input}, fc_count)

        return output


    def forward(self, input):
        return self._linear_forward(input)