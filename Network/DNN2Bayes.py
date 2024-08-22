import torch
import torch.nn as tn
import torch.nn as nn
from Network import ActFUnc_Module
import numpy as np


# This model is the original model from the codes of # B-PINNs (Bayesian Physics-Informed Neural Networks)
# This is the pytorch implementation of B-PINNs with Hamiltonian monte carlo algorithm.
# B-PINN에 관한 설명은 제 [블로그](https://www.notion.so/Physics-informed-neural-network-ee8cd5fa9ca243bfa5d7ce8d75370788) 에 있습니다.
class Net_2Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
        super(Net_2Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)
        # if init_W_B:
        #     tn.init.xavier_normal_(self.input_layer.weight)
        #     tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidden1.weight)
        #     tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.output_layer.weight)
        #     tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

        tn.init.xavier_normal_(self.input_layer.weight)
        tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)

        tn.init.xavier_normal_(self.hidden1.weight)
        tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)

        tn.init.xavier_normal_(self.output_layer.weight)
        tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)
        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
        super(Net_3Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[2], outdim, dtype=self.float_type, device=self.opt2device)

        # if init_W_B:
        #     tn.init.xavier_normal_(self.input_layer.weight)
        #     tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidden1.weight)
        #     tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidden2.weight)
        #     tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.output_layer.weight)
        #     tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

        # tn.init.xavier_normal_(self.input_layer.weight)
        # tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
        #
        # tn.init.xavier_normal_(self.hidden1.weight)
        # tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
        #
        # tn.init.xavier_normal_(self.hidden2.weight)
        # tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
        #
        # tn.init.xavier_normal_(self.output_layer.weight)
        # tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)
        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_4Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
        super(Net_4Hidden, self).__init__()
        self.layer_sizes = hidden_layer
        self.layer_list = []

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.input_layer = nn.Linear(indim, hidden_layer[0], dtype=self.float_type, device=self.opt2device)
        self.hidden1 = nn.Linear(hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.hidden2 = nn.Linear(hidden_layer[1], hidden_layer[2], dtype=self.float_type, device=self.opt2device)
        self.hidden3 = nn.Linear(hidden_layer[2], hidden_layer[3], dtype=self.float_type, device=self.opt2device)
        self.output_layer = nn.Linear(hidden_layer[3], outdim, dtype=self.float_type, device=self.opt2device)

        # if init_W_B:
        #     tn.init.xavier_normal_(self.input_layer.weight)
        #     tn.init.uniform_(self.input_layer.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidden1.weight)
        #     tn.init.uniform_(self.hidden1.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidden2.weight)
        #     tn.init.uniform_(self.hidden2.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.hidde3.weight)
        #     tn.init.uniform_(self.hidden3.bias, a=-1.0, b=1.0)
        #
        #     tn.init.xavier_normal_(self.output_layer.weight)
        #     tn.init.uniform_(self.output_layer.bias, a=-1.0, b=1.0)

    def forward(self, x):
        H_in = self.input_layer(x)
        H = self.actFunc_in(H_in)
        H = self.hidden1(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.hidden2(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.hidden3(H)  # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.output_layer(H)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_2Hidden_FourierBasis, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.Fourier_layer = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lout = tn.Linear(hidden_layer[1], outdim)

    def forward(self, x):
        H_FF = self.Fourier_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lout(x)  # Activation function is sin or tanh
        x = self.actFunc_out(x)
        return x


class Net_3Hidden_FourierBasis(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                type2float='float32', to_gpu=False, gpu_no=0):
        super(Net_3Hidden_FourierBasis, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_layer

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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.Fourier_layer = tn.Linear(indim, hidden_layer[0])
        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.l3 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.l4 = tn.Linear(hidden_layer[2], outdim)

    def forward(self, x):
        H_FF = self.Fourier_layer(x)
        H = torch.cat([torch.cos(H_FF), torch.sin(H_FF)], dim=-1)

        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l3(x)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.l4(x)
        x = self.actFunc_out(x)
        return x


class Net_2Hidden_FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.FF_layer = tn.Linear(indim, hidden_layer[0], bias=True, dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        # tn.init.uniform_(self.FF_layer.bias, a=-1.0 * sigma, b=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2ff
        self.FF_layer.bias.requires_grad = trainable2ff

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1], dtype=self.float_type, device=self.opt2device)
        self.lout = tn.Linear(hidden_layer[1], outdim, dtype=self.float_type, device=self.opt2device)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF = self.FF_layer(x)  # Fourier
        HFF = torch.cat([torch.cos(Hin_FF), torch.sin(Hin_FF)], dim=-1)
        H = self.lh1(HFF)          # Activation function is sin or tanh
        H = self.actFunc(H)
        H = self.lout(H)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
        super(Net_3Hidden_FF, self).__init__()
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

        self.layer_sizes = hidden_layer
        self.layer_list = []
        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.FF_layer = tn.Linear(indim, hidden_layer[0], bias=True, dtype=self.float_type, device=self.opt2device)
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        tn.init.uniform_(self.FF_layer.bias, a=-1.0 * sigma, b=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2ff
        self.FF_layer.bias.requires_grad = trainable2ff

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lh2 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.lout = tn.Linear(hidden_layer[2], outdim)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lh2.weight)
            tn.init.uniform_(self.lh2.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF = self.FF_layer(x)                                      # Fourier layer
        H = torch.cat([torch.cos(Hin_FF), torch.sin(Hin_FF)], dim=-1)  # Activation function is sin and cos
        x = self.lh1(H)                                                 # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lh2(x)                                                 # Activation function is sin or tanh
        x = self.actFunc(x)
        H = self.lout(x)                                                 # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_2FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0,
                 init_W_B=False):
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        # tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        # tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lout = tn.Linear(2*hidden_layer[1], outdim)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.lh1(HFF1)          # Activation function is sin or tanh
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.lh1(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        H_concat = torch.cat([H1, H2], dim=-1)

        H = self.lout(H_concat)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_2FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0,
                 init_W_B=False):
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.FF_layer1 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer1.weight, mean=0.0, std=1.0 * sigma1)
        # tn.init.uniform_(self.FF_layer1.bias, a=-1.0 * sigma1, b=1.0 * sigma1)
        self.FF_layer1.weight.requires_grad = trainable2ff
        self.FF_layer1.bias.requires_grad = trainable2ff

        self.FF_layer2 = tn.Linear(indim, hidden_layer[0], bias=True)
        tn.init.normal_(self.FF_layer2.weight, mean=0.0, std=1.0 * sigma2)
        # tn.init.uniform_(self.FF_layer2.bias, a=-1.0 * sigma2, b=1.0 * sigma2)
        self.FF_layer2.weight.requires_grad = trainable2ff
        self.FF_layer2.bias.requires_grad = trainable2ff

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lh2 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.lout = tn.Linear(2*hidden_layer[2], outdim)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lh2.weight)
            tn.init.uniform_(self.lh2.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.lh1(HFF1)  # Activation function is sin or tanh
        H1 = self.actFunc(H1)
        H1 = self.lh2(H1)
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.lh1(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)
        H2 = self.lh2(H2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        H_concat = torch.cat([H1, H2], dim=-1)

        H = self.lout(H_concat)  # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_2Hidden_3FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, sigma3=5.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0,
                 init_W_B=False):
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lout = tn.Linear(3*hidden_layer[1], outdim)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.lh1(HFF1)          # Activation function is sin or tanh
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.lh1(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        Hin_FF3 = self.FF_layer3(x)  # Fourier
        HFF3 = torch.cat([torch.cos(Hin_FF3), torch.sin(Hin_FF3)], dim=-1)
        H3 = self.lh1(HFF3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)

        H_concat = torch.cat([H1, H2, H3], dim=-1)

        H = self.lout(H_concat)            # Activation function is sin or tanh
        H = self.actFunc_out(H)
        return H


class Net_3Hidden_3FF(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 sigma1=1.0, sigma2=5.0, sigma3=10.0, trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0,
                 init_W_B=False):
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.lh1 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        self.lh2 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.lout = tn.Linear(3*hidden_layer[2], outdim)

        if init_W_B:
            tn.init.normal_(self.lh1.weight)
            tn.init.uniform_(self.lh1.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lh2.weight)
            tn.init.uniform_(self.lh2.bias, a=-1.0, b=1.0)

            tn.init.normal_(self.lout.weight)
            tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        Hin_FF1 = self.FF_layer1(x)  # Fourier
        HFF1 = torch.cat([torch.cos(Hin_FF1), torch.sin(Hin_FF1)], dim=-1)
        H1 = self.lh1(HFF1)  # Activation function is sin or tanh
        H1 = self.actFunc(H1)
        H1 = self.lh2(H1)
        H1 = self.actFunc(H1)

        Hin_FF2 = self.FF_layer2(x)  # Fourier
        HFF2 = torch.cat([torch.cos(Hin_FF2), torch.sin(Hin_FF2)], dim=-1)
        H2 = self.lh1(HFF2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)
        H2 = self.lh2(H2)  # Activation function is sin or tanh
        H2 = self.actFunc(H2)

        Hin_FF3 = self.FF_layer3(x)  # Fourier
        HFF3 = torch.cat([torch.cos(Hin_FF3), torch.sin(Hin_FF3)], dim=-1)
        H3 = self.lh1(HFF3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)
        H3 = self.lh2(H3)  # Activation function is sin or tanh
        H3 = self.actFunc(H3)

        H_concat = torch.cat([H1, H2, H3], dim=-1)

        H = self.lout(H_concat)  # Activation function is sin or tanh
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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


class Net_4Hidden_MultiScale(tn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, repeat_Highfreq=True, freq=None):
        super(Net_4Hidden_MultiScale, self).__init__()
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.lin = tn.Linear(indim, hidden_layer[0])
        self.lh1 = tn.Linear(hidden_layer[0], hidden_layer[1])
        self.lh2 = tn.Linear(hidden_layer[1], hidden_layer[2])
        self.lh3 = tn.Linear(hidden_layer[2], hidden_layer[3])
        self.lout = tn.Linear(hidden_layer[3], outdim)

    def forward(self, x):
        H = self.lin(x)
        H = self.actFunc_in(H * self.torch_mixcoe)
        x = self.lh1(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lh2(x)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lh3(x)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lout(x)  # Activation function is sin or tanh
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

        self.Fourier_layer = tn.Linear(indim, hidden_layer[0])
        tn.init.xavier_normal_(self.Fourier_layer.weight)
        tn.init.uniform_(self.Fourier_layer.bias, a=-1.0, b=1.0)

        self.l2 = tn.Linear(2*hidden_layer[0], hidden_layer[1])
        tn.init.xavier_normal_(self.l2.weight)
        tn.init.uniform_(self.l2.bias, a=-1.0, b=1.0)

        self.lout = tn.Linear(hidden_layer[1], outdim)
        tn.init.xavier_normal_(self.lout.weight)
        tn.init.uniform_(self.lout.bias, a=-1.0, b=1.0)

    def forward(self, x):
        H_FF = self.Fourier_layer(x)
        H = torch.cat([torch.cos(H_FF * self.torch_mixcoe), torch.sin(H_FF * self.torch_mixcoe)], dim=-1)
        x = self.l2(H)  # Activation function is sin or tanh
        x = self.actFunc(x)
        x = self.lout(x)  # Activation function is sin or tanh
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

        self.actFunc_in = ActFUnc_Module.my_actFunc(actName=actName2in)
        self.actFunc = ActFUnc_Module.my_actFunc(actName=actName)
        self.actFunc_out = ActFUnc_Module.my_actFunc(actName=actName2out)

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
