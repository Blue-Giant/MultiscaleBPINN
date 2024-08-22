# -*- coding: utf-8 -*-
"""
Created on 2021.06.16
modified on 2021.11.11
modified on 2022.11.11
Final version on 2023.05.01
@author: LXA
"""
import torch
import torch.nn as tn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt

"""
通常来说 torch.nn.functional 调用了 THNN 库，实现核心计算，但是不对 learnable_parameters 例如 weight bias ，进行管理，
为模型的使用带来不便。而 torch.nn 中实现的模型则对 torch.nn.functional，本质上是官方给出的对 torch.nn.functional的使用范例，
我们通过直接调用这些范例能够快速方便的使用 pytorch ，但是范例可能不能够照顾到所有人的使用需求，因此保留 torch.nn.functional 
来为这些用户提供灵活性，他们可以自己组装需要的模型。因此 pytorch 能够在灵活性与易用性上取得平衡。

特别注意的是，torch.nn不全都是对torch.nn.functional的范例，有一些调用了来自其他库的函数，例如常用的RNN型神经网络族即没有
在torch.nn.functional中出现。
参考链接：
        https://blog.csdn.net/gao2628688/article/details/99724617
"""


class my_actFunc(tn.Module):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def forward(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tnf.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tnf.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = torch.tanh(x_input)
        elif str.lower(self.actName) == 'enhance_tanh' or str.lower(self.actName) == 'enh_tanh':  # Enhance Tanh
            out_x = torch.tanh(0.5*torch.pi*x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tnf.relu(x_input)*tnf.relu(1-x_input)*torch.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tnf.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = torch.sin(x_input)
        elif str.lower(self.actName) == 'sinaddcos':
            out_x = 0.5*torch.sin(x_input) + 0.5*torch.cos(x_input)
            # out_x = 0.75*torch.sin(x_input) + 0.75*torch.cos(x_input)
            # out_x = torch.sin(x_input) + torch.cos(x_input)
        elif str.lower(self.actName) == 'fourier':
            out_x = torch.cat([torch.sin(x_input), torch.cos(x_input)], dim=-1)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tnf.sigmoid(x_input)
        elif str.lower(self.actName) == 'gelu':
            out_x = tnf.gelu(x_input)
        elif str.lower(self.actName) == 'gcu':
            out_x = x_input*torch.cos(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = tnf.mish(x_input)
        elif str.lower(self.actName) == 'silu':
            out_x = tnf.silu(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = torch.exp(-1.0 * x_input * x_input)
            # out_x = torch.exp(-0.5 * x_input * x_input)
        elif str.lower(self.actName) == 'requ':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'recu':
            out_x = tnf.relu(x_input)*tnf.relu(x_input)*tnf.relu(x_input)
        elif str.lower(self.actName) == 'morlet':
            out_x = torch.cos(1.75*x_input)*torch.exp(-0.5*x_input*x_input)
            # out_x = torch.cos(1.75 * x_input) * torch.exp(-1.0 * x_input * x_input)
        else:
            out_x = x_input
        return out_x


# ----------------dense net(constructing NN and initializing weights and bias )------------
class DenseNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(DenseNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        if str.lower(self.name2Model) == 'fourier_dnn':
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=False,
                                    dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(input_layer.weight)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                if i_layer == 0:
                    hidden_layer = tn.Linear(in_features=2 * hidden_units[i_layer],
                                             out_features=hidden_units[i_layer+1],
                                             dtype=self.float_type, device=self.opt2device)
                    tn.init.xavier_normal_(hidden_layer.weight)
                    tn.init.uniform_(hidden_layer.bias, -1, 1)
                else:
                    hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                             dtype=self.float_type, device=self.opt2device)
                    tn.init.xavier_normal_(hidden_layer.weight)
                    tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                                  dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, 0, 1)
            self.dense_layers.append(out_layer)
        else:
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0],
                                    dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, -1, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                         dtype=self.float_type, device=self.opt2device)
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                                  dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, -1, 1)
            self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.square(layer.weight))
                regular_b = regular_b + torch.sum(torch.square(layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def get_regular_sum2Fourier(self, regular_model='L2'):
        regular_w = 0.0
        regular_b = 0.0
        i_layer = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.abs(layer.bias))
                i_layer = i_layer + 1
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
                i_layer = i_layer + 1
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=0.5):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        # print(dense_in)
        H = dense_in(inputs)
        if str.lower(self.name2Model) == 'dnn':
            H = self.actFunc_in(H)
        else:
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            torch_mixcoe = torch.from_numpy(mixcoe)
            if self.to_gpu:
                torch_mixcoe = torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

            if str.lower(self.name2Model) == 'fourier_dnn':
                assert(self.actFunc_in.actName == 'fourier')
                H = sFourier*self.actFunc_in(H*torch_mixcoe)
            else:
                H = self.actFunc_in(H * torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            # print(dense_layer)
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        # print(dense_out)
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Dense_Net(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        if str.lower(self.name2Model) == 'fourier_dnn':
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0],
                                    dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, 0, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                if i_layer == 0:
                    hidden_layer = tn.Linear(in_features=2.0 * hidden_units[i_layer],
                                             out_features=hidden_units[i_layer+1], bias=False,
                                             dtype=self.float_type, device=self.opt2device)
                    tn.init.xavier_normal_(hidden_layer.weight)
                else:
                    hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                             dtype=self.float_type, device=self.opt2device)
                    tn.init.xavier_normal_(hidden_layer.weight)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                                  dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, 0, 1)
            self.dense_layers.append(out_layer)
        else:
            input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0],
                                    dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(input_layer.weight)
            tn.init.uniform_(input_layer.bias, -1, 1)
            self.dense_layers.append(input_layer)

            for i_layer in range(len(hidden_units)-1):
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                         dtype=self.float_type, device=self.opt2device)
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
                self.dense_layers.append(hidden_layer)

            out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                                  dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(out_layer.weight)
            tn.init.uniform_(out_layer.bias, -1, 1)
            self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.square(layer.weight))
                regular_b = regular_b + torch.sum(torch.square(layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        # print(dense_in)
        H = dense_in(inputs)
        if str.lower(self.name2Model) == 'fourier_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = torch.cat([torch.cos(H*mixcoe), torch.sin(H*mixcoe)], dim=-1)
        elif str.lower(self.name2Model) == 'scale_dnn' or str.lower(self.name2Model) == 'wavelet_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = self.actFunc_in(H*mixcoe)
        else:
            H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            # print(dense_layer)
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        # print(dense_out)
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------dense net(constructing NN and initializing weights and bias )------------
class Pure_DenseNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', to_gpu=False, gpu_no=0):
        super(Pure_DenseNet, self).__init__()
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0],
                                dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)
        H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        H = self.actFunc_out(H)
        return H


# ----------------Dense_ScaleNet(constructing NN and initializing weights and bias )------------
class Dense_ScaleNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0],
                                dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(input_layer.weight)
        tn.init.uniform_(input_layer.bias, -1, 1)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))
        H = self.actFunc_in(H*torch_mixcoe)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        out_results = self.actFunc_out(H)
        return out_results


# ----------------Fourier_FeatureDNN(constructing NN and initializing weights and bias )------------
class Fourier_FeatureDNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma=10.0, trainable2sigma=False):
        super(Fourier_FeatureDNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                  dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF = self.FF_layer(inputs)

        H = sFourier * torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H_out = dense_out(dense_out)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_2FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_2FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, trainable2sigma=False):
        super(Multi_2FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=2 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_3FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_3FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, trainable2sigma=False):
        super(Multi_3FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=3 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_4FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_4FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3,4: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, sigma4=10.0, trainable2sigma=False):
        super(Multi_4FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        self.FF_layer4 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer4.weight, mean=0.0, std=1.0 * sigma4)
        self.FF_layer4.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=4 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)
        H_FF4 = self.FF_layer4(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)
        H4 = sFourier * torch.cat([torch.cos(H_FF4), torch.sin(H_FF4)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)
            H4 = dense_layer(H4)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            H4 = self.actFunc(H4)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre

            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3, H4], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_5FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_5FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3,4,5: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, sigma4=10.0, sigma5=10.0,
                 trainable2sigma=False):
        super(Multi_5FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        self.FF_layer4 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer4.weight, mean=0.0, std=1.0 * sigma4)
        self.FF_layer4.weight.requires_grad = trainable2sigma

        self.FF_layer5 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer5.weight, mean=0.0, std=1.0 * sigma5)
        self.FF_layer5.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=5 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)
        H_FF4 = self.FF_layer4(inputs)
        H_FF5 = self.FF_layer5(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)
        H4 = sFourier * torch.cat([torch.cos(H_FF4), torch.sin(H_FF4)], dim=-1)
        H5 = sFourier * torch.cat([torch.cos(H_FF5), torch.sin(H_FF5)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)
            H4 = dense_layer(H4)
            H5 = dense_layer(H5)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            H4 = self.actFunc(H4)
            H5 = self.actFunc(H5)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre

            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3, H4, H5], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_8FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_6FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3,4,5,6,7,8: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, sigma4=10.0, sigma5=10.0, sigma6=10.0,
                 trainable2sigma=False):
        super(Multi_6FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        self.FF_layer4 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer4.weight, mean=0.0, std=1.0 * sigma4)
        self.FF_layer4.weight.requires_grad = trainable2sigma

        self.FF_layer5 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer5.weight, mean=0.0, std=1.0 * sigma5)
        self.FF_layer5.weight.requires_grad = trainable2sigma

        self.FF_layer6 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer6.weight, mean=0.0, std=1.0 * sigma6)
        self.FF_layer6.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=6 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)
        H_FF4 = self.FF_layer4(inputs)
        H_FF5 = self.FF_layer5(inputs)
        H_FF6 = self.FF_layer6(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)
        H4 = sFourier * torch.cat([torch.cos(H_FF4), torch.sin(H_FF4)], dim=-1)
        H5 = sFourier * torch.cat([torch.cos(H_FF5), torch.sin(H_FF5)], dim=-1)
        H6 = sFourier * torch.cat([torch.cos(H_FF6), torch.sin(H_FF6)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5
            H6_pre = H6

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)
            H4 = dense_layer(H4)
            H5 = dense_layer(H5)
            H6 = dense_layer(H6)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            H4 = self.actFunc(H4)
            H5 = self.actFunc(H5)
            H6 = self.actFunc(H6)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre
                H6 = H6 + H6_pre

            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3, H4, H5, H6], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_8FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_7FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3,4,5,6,7,8: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, sigma4=10.0, sigma5=10.0, sigma6=10.0,
                 sigma7=10.0, trainable2sigma=False):
        super(Multi_7FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        self.FF_layer4 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer4.weight, mean=0.0, std=1.0 * sigma4)
        self.FF_layer4.weight.requires_grad = trainable2sigma

        self.FF_layer5 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer5.weight, mean=0.0, std=1.0 * sigma5)
        self.FF_layer5.weight.requires_grad = trainable2sigma

        self.FF_layer6 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer6.weight, mean=0.0, std=1.0 * sigma6)
        self.FF_layer6.weight.requires_grad = trainable2sigma

        self.FF_layer7 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer7.weight, mean=0.0, std=1.0 * sigma7)
        self.FF_layer7.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=7 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)
        H_FF4 = self.FF_layer4(inputs)
        H_FF5 = self.FF_layer5(inputs)
        H_FF6 = self.FF_layer6(inputs)
        H_FF7 = self.FF_layer7(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)
        H4 = sFourier * torch.cat([torch.cos(H_FF4), torch.sin(H_FF4)], dim=-1)
        H5 = sFourier * torch.cat([torch.cos(H_FF5), torch.sin(H_FF5)], dim=-1)
        H6 = sFourier * torch.cat([torch.cos(H_FF6), torch.sin(H_FF6)], dim=-1)
        H7 = sFourier * torch.cat([torch.cos(H_FF7), torch.sin(H_FF7)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5
            H6_pre = H6
            H7_pre = H7

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)
            H4 = dense_layer(H4)
            H5 = dense_layer(H5)
            H6 = dense_layer(H6)
            H7 = dense_layer(H7)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            H4 = self.actFunc(H4)
            H5 = self.actFunc(H5)
            H6 = self.actFunc(H6)
            H7 = self.actFunc(H7)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre
                H6 = H6 + H6_pre
                H7 = H7 + H7_pre

            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3, H4, H5, H6, H7], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_8FF_DNN(constructing NN and initializing weights and bias )------------
class Multi_8FF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam1,2,3,4,5,6,7,8: the factor of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigma1=10.0, sigma2=10.0, sigma3=10.0, sigma4=10.0, sigma5=10.0, sigma6=10.0,
                 sigma7=10.0, sigma8=10.0, trainable2sigma=False):
        super(Multi_8FF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        self.FF_layer1 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2sigma

        self.FF_layer2 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2sigma

        self.FF_layer3 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2sigma

        self.FF_layer4 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer4.weight, mean=0.0, std=1.0 * sigma4)
        self.FF_layer4.weight.requires_grad = trainable2sigma

        self.FF_layer5 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer5.weight, mean=0.0, std=1.0 * sigma5)
        self.FF_layer5.weight.requires_grad = trainable2sigma

        self.FF_layer6 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer6.weight, mean=0.0, std=1.0 * sigma6)
        self.FF_layer6.weight.requires_grad = trainable2sigma

        self.FF_layer7 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer7.weight, mean=0.0, std=1.0 * sigma7)
        self.FF_layer7.weight.requires_grad = trainable2sigma

        self.FF_layer8 = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                   dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer8.weight, mean=0.0, std=1.0 * sigma8)
        self.FF_layer8.weight.requires_grad = trainable2sigma

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=8 * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        H_FF1 = self.FF_layer1(inputs)
        H_FF2 = self.FF_layer2(inputs)
        H_FF3 = self.FF_layer3(inputs)
        H_FF4 = self.FF_layer4(inputs)
        H_FF5 = self.FF_layer5(inputs)
        H_FF6 = self.FF_layer6(inputs)
        H_FF7 = self.FF_layer7(inputs)
        H_FF8 = self.FF_layer8(inputs)

        H1 = sFourier * torch.cat([torch.cos(H_FF1), torch.sin(H_FF1)], dim=-1)
        H2 = sFourier * torch.cat([torch.cos(H_FF2), torch.sin(H_FF2)], dim=-1)
        H3 = sFourier * torch.cat([torch.cos(H_FF3), torch.sin(H_FF3)], dim=-1)
        H4 = sFourier * torch.cat([torch.cos(H_FF4), torch.sin(H_FF4)], dim=-1)
        H5 = sFourier * torch.cat([torch.cos(H_FF5), torch.sin(H_FF5)], dim=-1)
        H6 = sFourier * torch.cat([torch.cos(H_FF6), torch.sin(H_FF6)], dim=-1)
        H7 = sFourier * torch.cat([torch.cos(H_FF7), torch.sin(H_FF7)], dim=-1)
        H8 = sFourier * torch.cat([torch.cos(H_FF8), torch.sin(H_FF8)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units) - 1):
            H1_pre = H1
            H2_pre = H2
            H3_pre = H3
            H4_pre = H4
            H5_pre = H5
            H6_pre = H6
            H7_pre = H7
            H8_pre = H8

            dense_layer = self.dense_layers[i_layer]

            H1 = dense_layer(H1)
            H2 = dense_layer(H2)
            H3 = dense_layer(H3)
            H4 = dense_layer(H4)
            H5 = dense_layer(H5)
            H6 = dense_layer(H6)
            H7 = dense_layer(H7)
            H8 = dense_layer(H8)

            H1 = self.actFunc(H1)
            H2 = self.actFunc(H2)
            H3 = self.actFunc(H3)
            H4 = self.actFunc(H4)
            H5 = self.actFunc(H5)
            H6 = self.actFunc(H6)
            H7 = self.actFunc(H7)
            H8 = self.actFunc(H8)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H1 = H1 + H1_pre
                H2 = H2 + H2_pre
                H3 = H3 + H3_pre
                H4 = H4 + H4_pre
                H5 = H5 + H5_pre
                H6 = H6 + H6_pre
                H7 = H7 + H7_pre
                H8 = H8 + H8_pre

            hidden_record = self.hidden_units[i_layer + 1]

        H_concat = torch.cat([H1, H2, H3, H4, H5, H6, H7, H8], dim=-1)
        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Multi_NFF_DNN(constructing NN and initializing weights and bias )------------
class Multi_NFF_DNN(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
        sigam_vec: the vector of sigma for Fourier input feature embedding
        trainable2sigma: train the sigma matrix or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_Highfreq=True, type2float='float32',
                 to_gpu=False, gpu_no=0, sigam_vec=None, trainable2sigma=False):
        super(Multi_NFF_DNN, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()
        self.dense_FF_layers = tn.ModuleList()
        self.num2sigma = len(sigam_vec)
        self.mat2out = []

        for i_ff in range(self.num2sigma):
            FF_layer = tn.Linear(in_features=indim, out_features=hidden_units[0] // 2, bias=False,
                                 dtype=self.float_type, device=self.opt2device)
            tn.init.normal_(FF_layer.weight, mean=0.0, std=1.0) * float(sigam_vec[i_ff])
            FF_layer.weight.requires_grad = trainable2sigma
            self.dense_FF_layers.append(FF_layer)

            zeros_vec = np.zeros(shape=(hidden_units[-1], len(sigam_vec) * hidden_units[-1]))
            zeros_vec[:, i_ff*hidden_units[-1]:(i_ff+1)*hidden_units[-1]] = 1
            zeros_ones = torch.tensor(zeros_vec, dtype=self.float_type, device=self.opt2device)
            self.mat2out.append(zeros_ones)

        for i_layer in range(len(hidden_units) - 1):
            hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer + 1],
                                     dtype=self.float_type, device=self.opt2device)
            tn.init.xavier_normal_(hidden_layer.weight)
            tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=len(sigam_vec) * hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0
        regular_b = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                regular_b = regular_b + torch.sum(torch.abs(layer.bias))
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        H_concat = 0
        for i_ff in range(self.num2sigma):
            # ------ dealing with the input data ---------------
            ff_embedding = self.dense_FF_layers[i_ff]
            H_FF = ff_embedding(inputs)
            H = sFourier * torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

            #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
            hidden_record = self.hidden_units[0]
            for i_layer in range(0, len(self.hidden_units) - 1):
                H_pre = H
                dense_layer = self.dense_layers[i_layer]
                H = dense_layer(H)
                H = self.actFunc(H)
                if self.hidden_units[i_layer + 1] == hidden_record:
                    H = H + H_pre
                hidden_record = self.hidden_units[i_layer + 1]

            assemble_Vec2H = self.mat2out[i_ff]
            H_concat = H_concat + torch.matmul(H, assemble_Vec2H)

        dense_out = self.dense_layers[-1]
        H_out = dense_out(H_concat)
        out_results = self.actFunc_out(H_out)
        return out_results


# ----------------Dense Fourier—Net(constructing NN and initializing weights and bias )------------
class Dense_FourierNet(tn.Module):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_Highfreq: repeating the high-frequency component of scale-transformation factor or not
        type2float: the numerical type
        to_gpu: using GPU or not
        gpu_no: if the GPU is required, the no of GPU
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='tanh', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', type2float='float32', to_gpu=False, gpu_no=0,
                 repeat_Highfreq=True):
        super(Dense_FourierNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.repeat_Highfreq = repeat_Highfreq
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        if type2float == 'float32':
            self.float_type = torch.float32
        elif type2float == 'float64':
            self.float_type = torch.float64
        elif type2float == 'float16':
            self.float_type = torch.float16

        if to_gpu:
            self.opt2device = 'cuda:' + str(gpu_no)
        else:
            self.opt2device = 'cpu'

        self.dense_layers = tn.ModuleList()

        input_layer = tn.Linear(in_features=indim, out_features=hidden_units[0], bias=False,
                                dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(input_layer.weight)
        self.dense_layers.append(input_layer)

        for i_layer in range(len(hidden_units)-1):
            if i_layer == 0:
                hidden_layer = tn.Linear(in_features=2 * hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                         dtype=self.float_type, device=self.opt2device)
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            else:
                hidden_layer = tn.Linear(in_features=hidden_units[i_layer], out_features=hidden_units[i_layer+1],
                                         dtype=self.float_type, device=self.opt2device)
                tn.init.xavier_normal_(hidden_layer.weight)
                tn.init.uniform_(hidden_layer.bias, -1, 1)
            self.dense_layers.append(hidden_layer)

        out_layer = tn.Linear(in_features=hidden_units[-1], out_features=outdim,
                              dtype=self.float_type, device=self.opt2device)
        tn.init.xavier_normal_(out_layer.weight)
        tn.init.uniform_(out_layer.bias, -1, 1)
        self.dense_layers.append(out_layer)

    def get_regular_sum2WB(self, regular_model='L0'):
        regular_w = 0.0
        regular_b = 0.0
        i_layer = 0
        if regular_model == 'L1':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.abs(layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.abs(layer.bias))
                i_layer = i_layer + 1
        elif regular_model == 'L2':
            for layer in self.dense_layers:
                regular_w = regular_w + torch.sum(torch.mul(layer.weight, layer.weight))
                if i_layer != 0:
                    regular_b = regular_b + torch.sum(torch.mul(layer.bias, layer.bias))
                i_layer = i_layer + 1
        else:
            regular_w = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
            regular_b = torch.tensor(0.0, dtype=self.float_type, device=self.opt2device)
        return regular_w + regular_b

    def forward(self, inputs, scale=None, sFourier=1.0, training=None, mask=None):
        # ------ dealing with the input data ---------------
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)

        Unit_num = int(self.hidden_units[0] / len(scale))
        mixcoe = np.repeat(scale, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            torch_mixcoe = torch_mixcoe.cuda(device='cuda:'+str(self.gpu_no))

        H = sFourier*torch.cat([torch.cos(H*torch_mixcoe), torch.sin(H*torch_mixcoe)], dim=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(0, len(self.hidden_units)-1):
            H_pre = H
            dense_layer = self.dense_layers[i_layer+1]
            H = dense_layer(H)
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        dense_out = self.dense_layers[-1]
        H = dense_out(H)
        out_results = self.actFunc_out(H)
        return out_results



