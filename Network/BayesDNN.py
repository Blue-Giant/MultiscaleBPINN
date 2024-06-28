import torch
import torch.nn as tn
import torch.nn as nn
from Network import ActFUnc_Module


# This model is the original model from the codes of # B-PINNs (Bayesian Physics-Informed Neural Networks)
# This is the pytorch implementation of B-PINNs with Hamiltonian monte carlo algorithm.
# B-PINN에 관한 설명은 제 [블로그](https://www.notion.so/Physics-informed-neural-network-ee8cd5fa9ca243bfa5d7ce8d75370788) 에 있습니다.
class Net_2Hidden(nn.Module):
    def __init__(self, indim=1, outdim=1, hidden_layer=None, actName2in='tanh', actName='tanh', actName2out='linear',
                 type2float='float32', to_gpu=False, gpu_no=0, init_W_B=False):
        super(Net_2Hidden, self).__init__()
        self.layer_sizes = hidden_layer

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
