import torch
import torch.nn as tn
import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np


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


# This model is the original model from the codes of # B-PINNs (Bayesian Physics-Informed Neural Networks)
# This is the pytorch implementation of B-PINNs with Hamiltonian monte carlo algorithm.
# B-PINN에 관한 설명은 제 [블로그](https://www.notion.so/Physics-informed-neural-network-ee8cd5fa9ca243bfa5d7ce8d75370788) 에 있습니다.
class Net_2Hidden(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh, sigma=5.0, trainable2sigma=False, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Net_2Hidden, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

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

        self.FF_layer = nn.Linear(layer_sizes[0], layer_sizes[1], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.FF_layer.bias, a=-1.0, b=1.0)
        self.FF_layer.weight.requires_grad = trainable2sigma

        self.l2 = nn.Linear(2*layer_sizes[1], layer_sizes[2], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.l2.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.l2.bias, a=-1.0, b=1.0)

        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.l3.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.l3.bias, a=-1.0, b=1.0)
        # self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])

    def forward(self, x):
        H_FF = self.FF_layer(x)  # Fourier
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.activation(x)
        x = self.l3(x)  # Activation function is sin or tanh
        # x = self.activation(x)
        # x = self.l4(x)   # Activation function is linear
        return x


class Net_3Hidden(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh, sigma=5.0, trainable2sigma=False, type2float='float32',
                 to_gpu=False, gpu_no=0):
        super(Net_3Hidden, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

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

        self.FF_layer = nn.Linear(layer_sizes[0], layer_sizes[1], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.FF_layer.bias, a=-1.0, b=1.0)
        self.FF_layer.weight.requires_grad = trainable2sigma

        self.l2 = nn.Linear(2 * layer_sizes[1], layer_sizes[2], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.l2.weight, mean=0.0, std=1.0)
        tn.init.uniform_(self.l2.bias, a=-1.0, b=1.0)

        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3], dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.l3.weight, mean=0.0, std=1.0)
        tn.init.uniform_(self.l3.bias, a=-1.0, b=1.0)

        self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        tn.init.normal_(self.l4.weight, mean=0.0, std=1.0)
        tn.init.uniform_(self.l4.bias, a=-1.0, b=1.0)

    def forward(self, x):
        H_FF = self.FF_layer(x)  # Fourier
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.activation(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.activation(x)
        x = self.l4(x)   # Activation function is linear
        return x


class Net_2Hidden_FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_2Hidden_FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

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

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer = tn.Linear(indim, hidden_layer[0], bias=True, dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.FF_layer.bias, a=-1.0 * sigma, b=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2ff
        self.FF_layer.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.l3 = tn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

    def forward(self, x):
        Hin_FF = self.FF_layer(x)  # Fourier
        HFF = torch.cat([torch.cos(Hin_FF), torch.sin(Hin_FF)], dim=-1)
        H = self.l2(HFF)          # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.l3(H)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_3Hidden_FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.FF_layer.bias, a=-1.0 * sigma, b=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2ff
        self.FF_layer.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(hidden_layer[2], outdim)

    def forward(self, x):
        Hin_FF = self.FF_layer(x)                                      # Fourier layer
        H = torch.cat([torch.cos(Hin_FF), torch.sin(Hin_FF)], dim=-1)  # Activation function is sin and cos
        x = self.l2(H)                                                 # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)                                                 # Activation function is sin or tanh
        x = self.actFunc(x)
        H = self.l4(x)                                                 # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_2FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_2Hidden_2FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

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

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(2*hidden_layer[1], outdim)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.l2(HFF1)          # Activation function is sin or tanh
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.l2(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        H_concat = torch.cat([H1, H2], dim=-1)

        H = self.l3(H_concat)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_2FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_3Hidden_2FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

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

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(2*hidden_layer[2], outdim)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.l2(HFF1)  # Activation function is sin or tanh
        H1 = self.actFunc(H1)
        H1 = self.l3(H1)
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.l2(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)
        H2 = self.l3(H2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        H_concat = torch.cat([H1, H2], dim=-1)

        H = self.l4(H_concat)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_3FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, sigma3=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_2Hidden_3FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

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

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.FF_layer3 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        tn.init.uniform_(self.FF_layer3.bias, a=-1.0 * sigma3, b=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2ff
        self.FF_layer3.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(3*hidden_layer[1], outdim)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.l2(HFF1)          # Activation function is sin or tanh
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.l2(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        Hin_FF3 = self.FF_layer3(x)  # Fourier
        HFF3 = torch.cat([torch.cos(Hin_FF3), torch.sin(Hin_FF3)], dim=-1)
        H3 = self.l2(HFF3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)

        H_concat = torch.cat([H1, H2, H3], dim=-1)

        H = self.l3(H_concat)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_3FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, sigma3=10.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_3Hidden_3FF, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

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

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.FF_layer3 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer3.weight, mean=0.0, std=1.0 * sigma3)
        tn.init.uniform_(self.FF_layer3.bias, a=-1.0 * sigma3, b=1.0 * sigma3)
        self.FF_layer3.weight.requires_grad = trainable2ff
        self.FF_layer3.bias.requires_grad = trainable2ff

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(3*hidden_layer[2], outdim)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.l2(HFF1)  # Activation function is sin or tanh
        H1 = self.actFunc(H1)
        H1 = self.l3(H1)
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.l2(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)
        H2 = self.l3(H2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        Hin_FF3 = self.FF_layer3(x)  # Fourier
        HFF3 = torch.cat([torch.cos(Hin_FF3), torch.sin(Hin_FF3)], dim=-1)
        H3 = self.l2(HFF3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)
        H3 = self.l3(H3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)

        H_concat = torch.cat([H1, H2, H3], dim=-1)

        H = self.l4(H_concat)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None):
        super(Net_2Hidden_MultiScale, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.repeat_Highfreq = repeat_Highfreq
        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

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

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.layer_list = []

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.l1 = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], outdim)

    def forward(self, x):
        H = self.l1(x)
        H = self.actFunc_in(H * self.torch_mixcoe)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.actFunc_out(x)
        return x


class Net_3Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None):
        super(Net_3Hidden_MultiScale, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.repeat_Highfreq = repeat_Highfreq
        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

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

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.layer_list = []

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.l1 = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(hidden_layer[2], outdim)

    def forward(self, x):
        H = self.l1(x)
        H = self.actFunc_in(H * self.torch_mixcoe)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l4(x)  # Activation function is sin or tanh
        x = self.actFunc_out(x)
        return x


class Net_2Hidden_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None):
        super(Net_2Hidden_FourierBase, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.repeat_Highfreq = repeat_Highfreq
        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

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

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.layer_list = []

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.Fourier_layer = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], outdim)

    def forward(self, x):
        H_FF = self.Fourier_layer(x)
        H = torch.cat([torch.cos(H_FF * self.torch_mixcoe), torch.sin(H_FF * self.torch_mixcoe)], dim=-1)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.actFunc_out(x)
        return x


class Net_3Hidden_FourierBase(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None):
        super(Net_3Hidden_FourierBase, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

        self.repeat_Highfreq = repeat_Highfreq
        self.scales = freq
        self.to_gpu = to_gpu
        self.gpu_no = gpu_no

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

        Unit_num = int(self.hidden_units[0] / len(freq))
        mixcoe = np.repeat(freq, Unit_num)

        if self.repeat_Highfreq == True:
            mixcoe = np.concatenate(
                (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[-1]))
        else:
            mixcoe = np.concatenate(
                (np.ones([self.hidden_units[0] - Unit_num * len(freq)]) * freq[0], mixcoe))

        mixcoe = mixcoe.astype(np.float32)
        self.torch_mixcoe = torch.from_numpy(mixcoe)
        if self.to_gpu:
            self.torch_mixcoe = self.torch_mixcoe.cuda(device='cuda:' + str(self.gpu_no))

        self.layer_list = []

        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)

        self.Fourier_layer = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(hidden_layer[2], outdim)

    def forward(self, x):
        H_FF = self.Fourier_layer(x)
        H = torch.cat([torch.cos(H_FF * self.torch_mixcoe), torch.sin(H_FF * self.torch_mixcoe)], dim=-1)

        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l4(x)
        x = self.actFunc_out(x)
        return x