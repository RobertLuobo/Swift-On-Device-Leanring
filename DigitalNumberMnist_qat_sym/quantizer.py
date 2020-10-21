import torch
import torch.nn.functional as F
from collections import namedtuple
from config import cfg

QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])

'''
Following functions for symmetric Mapping-[-128 ~ +127] 
'''

def calcScaleZeroPointSym(min_val, max_val, num_bits=8):
    # Calc Scale
    max_val = max(abs(min_val), abs(max_val))
    qmax = 2. ** (num_bits - 1) - 1.
    qmin = -qmax
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = 0
    return scale, zero_point


n_count = 0
input_n_count = 1


def quantize_tensor_input_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmax = 2. ** (num_bits - 1) - 1.


    scale, zero_point = calcScaleZeroPointSym(min_val, max_val, num_bits)
    q_x = x / scale
    # q_x.clamp_(-qmax - 1, qmax).round_()
    q_x = q_x.round()

    global input_n_count
    cfg.writer.add_histogram("input", x, n_count)
    cfg.writer.add_scalars("input_scale", {"input_scale": scale}, n_count)
    # cfg.writer.add_scalars("input_zero_point", {"input_zero_point": zero_point}, n_count)
    input_n_count += 1

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


weight_n_count = 0


def quantize_tensor_weight_sym(x, num_bits=8, min_val=None, max_val=None):
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    qmax = 2. ** (num_bits - 1) - 1.

    scale, zero_point = calcScaleZeroPointSym(min_val, max_val, num_bits)
    q_x = x / scale
    # q_x.clamp_(-qmax - 1, qmax).round_()
    q_x = q_x.round()

    global weight_n_count
    cfg.writer.add_histogram("weight", x, n_count)
    cfg.writer.add_scalars("weight_scale", {"weight_scale": scale}, n_count)
    # cfg.writer.add_scalars("weight_zero_point", {"weight_zero_point": zero_point}, n_count)
    weight_n_count += 1

    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)


def dequantize_tensor_sym(q_x):
    return q_x.scale * (q_x.tensor.float())



Conv_count = 0
fc_count = 0

'''
Dong Xin paper uniform quantizer method: alpha * {0 , 1/127, -1/127, .... +1, -1 } by APoT
formula:   
'''

class Qconv2d_INT(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        torch.nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride=stride,
                                 padding=padding, dilation=dilation, groups=groups, bias=bias)

    def conv2d_forward(self, input, num_bits=8):
        # step1 : discrete
        min_val = torch.min(input)
        max_val = torch.max(input)
        q_input = quantize_tensor_input_sym(input, num_bits=num_bits, min_val=min_val, max_val=max_val)

        q_weight = quantize_tensor_weight_sym(self.weight, num_bits=num_bits,
                                              min_val=torch.min(self.weight),
                                              max_val=torch.max(self.weight)
                                          )

        if(self.bias != None):
            bias = self.bias.view(self.out_channels,1,1)

        int_x = q_input.tensor.type(torch.IntTensor)
        int_w = q_weight.tensor.type(torch.IntTensor)
        output = torch.nn.functional.conv2d(
                        int_x,int_w,
                        bias=None,
                        stride=self.stride, padding=self.padding,
                        dilation=self.dilation, groups=self.groups
                )
        # print(bias.size(), output.size())
        # step2 : find out alpha

        # input_alpha = (input.max() - input.min())
        # weight_alpha = (self.weight.max() - self.weight.min())
        #
        # input_alpha = (input.max() - input.min())
        # weight_alpha = (self.weight.max() - self.weight.min())

        '''Test quanti-value and dequanti-value'''
        fp32_x = dequantize_tensor_sym(
                    QTensor(
                            tensor=int_x,
                            scale=q_input.scale,
                            zero_point=0
                        )
                    )
        fp32_w = dequantize_tensor_sym(
                    QTensor(
                            tensor=int_w,
                            scale=q_weight.scale,
                            zero_point=0
                        )
                    )
        # print("Qconv2d_INT input delta", torch.sum(input - fp32_x),
        #       "\tweight delta：",torch.sum(self.weight - fp32_w))
        global Conv_count

        Conv_count += 1
        # cfg.writer.add_scalars("Conv_input_alpha", {"Conv_input_alpha": input_alpha}, Conv_count)
        # cfg.writer.add_scalars("Conv_weight_alpha", {"Conv_weight_alpha": input_alpha}, Conv_count)

        # Step3 : calibration, add decimal point
        # decimal_point = 2 ** (num_bits - 1) - 1
        # output = output / (decimal_point ** 2) * input_alpha * weight_alpha
        # output = output * (q_weight.scale * q_input.scale)
        # Step4 :  dequantizer

        Qoutput = QTensor(
                    tensor=output,
                    scale=q_weight.scale * q_input.scale,
                    zero_point=0
                    )

        output = dequantize_tensor_sym(Qoutput)
        if (self.bias != None):
            output = output + bias

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
        q_input = quantize_tensor_input_sym(input, num_bits=num_bits,
                                            min_val=torch.min(input), max_val=torch.max(input))

        q_weight = quantize_tensor_weight_sym(self.weight, num_bits=num_bits,
                                              min_val=torch.min(self.weight), max_val=torch.max(self.weight))

        if(self.bias != None):
            bias = self.bias

        int_x = q_input.tensor.type(torch.IntTensor)
        int_w = q_weight.tensor.type(torch.IntTensor)

        output = torch.nn.functional.linear(int_x, int_w, bias = None)
        # step2 : find out alpha
        # input_alpha = (input.max() - input.min())
        # weight_alpha = (self.weight.max() - self.weight.min())
        '''Test quanti-value and dequanti-value'''
        fp32_x = dequantize_tensor_sym(
                    QTensor(
                        tensor=int_x,
                        scale=q_input.scale,
                        zero_point=0
                        )
                )
        fp32_w = dequantize_tensor_sym(
                    QTensor(
                        tensor=int_w,
                        scale=q_weight.scale,
                        zero_point=0
                        )
                )

        # print("QLinear_INT input delta", torch.sum(input - fp32_x), "\tweight delta：", torch.sum((self.weight - fp32_w)))
        global Conv_count
        Conv_count += 1
        # cfg.writer.add_scalars("Conv_input_alpha", {"Conv_input_alpha": input_alpha}, Conv_count)
        # cfg.writer.add_scalars("Conv_weight_alpha", {"Conv_weight_alpha": input_alpha}, Conv_count)

        # Step3 : calibration, add decimal point
        # decimal_point = 2 ** (num_bits - 1) - 1
        # output = output / (decimal_point ** 2) * input_alpha * weight_alpha

        # Step4 :  dequantizer

        Qoutput = QTensor(
                tensor=output,
                scale=q_weight.scale * q_input.scale,
                zero_point=0
               )

        output = dequantize_tensor_sym(Qoutput)

        if(self.bias != None):
            output = output + bias

        return output

    def forward(self, input):
        return self._linear_forward(input)


'''
Following functions for Asymmetric Mapping-[0-255] 
'''

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


def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)


class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output, num_bits=8, min_val=None, max_val=None):
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
        output = torch.nn.functional.conv2d(FakeQuantOp.apply(input), FakeQuantOp.apply(self.weight), self.bias,
                                            self.stride, self.padding,
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

