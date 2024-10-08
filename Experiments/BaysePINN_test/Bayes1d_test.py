# %%

import torch
from Network import hamiltorch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from Network import util
import torch
import torch.nn as tn
import torch.nn.functional as tnf
from torch.nn.parameter import Parameter
from Network.DNN_base import my_actFunc

# device

print(f"Is CUDA available?: {torch.cuda.is_available()}")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

with_gpu = True
if "cpu" == str.lower(device):
    with_gpu = False

# hyperparameters

hamiltorch.set_random_seed(123)
prior_std = 15
like_std = 0.1
step_size = 0.001
burn = 100
num_samples = 200
L = 100
layer_sizes = [1, 16, 20, 1]
activation = torch.sin
pde = True
pinns = False
epochs = 10000
tau_priors = 1 / prior_std**2
tau_likes = 1 / like_std**2

lb = 0
ub = 1
N_tr_u = 50
N_tr_f = 50
N_tr_k = 0
N_val = 100


def u(x):
    return torch.sin(2 * np.pi * x) + 0.5*torch.cos(10*np.pi * x)


def k(x):
    return 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15**2)


def f(x):
    return 0.01 * (-4 * np.pi**2 * torch.sin(2 * np.pi * x) - 50 * np.pi**2 * torch.cos(10*np.pi*x)) + k(x) * u(x)


data = {}
data["x_u"] = torch.cat(
    (torch.linspace(lb, ub, 2).view(-1, 1), (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb)
)  # torch.linspace(start, end, steps) view making it a column vector, boundary points
data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias
data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb   # interior points
data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std
data["x_k"] = (ub - lb) * torch.rand(N_tr_k, 1) + lb
data["y_k"] = k(data["x_k"]) + torch.randn_like(data["x_k"]) * like_std

data_val = {}  # real value
data_val["x_u"] = torch.linspace(lb, ub, N_val).view(-1, 1)
data_val["y_u"] = u(data_val["x_u"])
data_val["x_f"] = torch.linspace(lb, ub, N_val).view(-1, 1)
data_val["y_f"] = f(data_val["x_f"])
data_val["x_k"] = torch.linspace(lb, ub, N_val).view(-1, 1)
data_val["y_k"] = k(data_val["x_k"])

for d in data:
    data[d] = data[d].to(device)
for d in data_val:
    data_val[d] = data_val[d].to(device)

# model


class Net(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh, sigma=5.0, trainable2sigma=False):
        super(Net, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer_list = []
        self.activation = activation

        self.FF_layer = nn.Linear(layer_sizes[0], layer_sizes[1])
        tn.init.normal_(self.FF_layer.weight, mean=0.0, std=1.0 * sigma)
        self.FF_layer.weight.requires_grad = trainable2sigma

        self.l2 = nn.Linear(2*layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
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


net_u = Net(layer_sizes, activation, sigma=15.0, trainable2sigma=True).to(device)
net_k = Net(layer_sizes, activation, sigma=5.0, trainable2sigma=True).to(device)

# net_u = Fourier_FeatureDNN(indim=1, outdim=1, hidden_units=layer_sizes, actName2in='sin', actName='sin',
#                            actName2out='linear', repeat_Highfreq=False, type2float='float32', to_gpu=with_gpu,
#                            gpu_no=0, sigma=5.0).to(device)
# net_k = Fourier_FeatureDNN(indim=1, outdim=1, hidden_units=layer_sizes, actName2in='sin', actName='sin',
#                            actName2out='linear', repeat_Highfreq=False, type2float='float32', to_gpu=with_gpu,
#                            gpu_no=0, sigma=5.0).to(device)
nets = [net_u, net_k]


def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_k = data["x_k"]
    y_k = data["y_k"]
    pred_k = fmodel[1](x_k, params=params_unflattened[1])
    ll = ll - 0.5 * tau_likes[1] * ((pred_k - y_k) ** 2).sum(0)
    x_f = data["x_f"]
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]
    k = fmodel[1](x_f, params=params_unflattened[1])
    pred_f = 0.01 * u_xx + k * u
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[2] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_k, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, k, pred_u, pred_k, pred_f
        torch.cuda.empty_cache()

    return ll, output


# sampling

params_hmc = util.sample_model_bpinns(
    nets,
    data,
    model_loss=model_loss,
    num_samples=num_samples,
    num_steps_per_sample=L,
    step_size=step_size,
    burn=burn,
    tau_priors=tau_priors,
    tau_likes=tau_likes,
    device=device,
    pde=pde,
    pinns=pinns,
    epochs=epochs,
)

pred_list, log_prob_list = util.predict_model_bpinns(
    nets,
    params_hmc,
    data_val,
    model_loss=model_loss,
    tau_priors=tau_priors,
    tau_likes=tau_likes,
    pde=pde,
)

print(
    "\nExpected validation log probability: {:.3f}".format(
        torch.stack(log_prob_list).mean()
    )
)

pred_list_u = pred_list[0].cpu().numpy()
pred_list_k = pred_list[1].cpu().numpy()
pred_list_f = pred_list[2].cpu().numpy()

# plot

x_val = data_val["x_u"].cpu().numpy()
u_val = data_val["y_u"].cpu().numpy()
k_val = data_val["y_k"].cpu().numpy()
f_val = data_val["y_f"].cpu().numpy()
x_u = data["x_u"].cpu().numpy()
y_u = data["y_u"].cpu().numpy()
x_f = data["x_f"].cpu().numpy()
y_f = data["y_f"].cpu().numpy()
x_k = data["x_k"].cpu().numpy()
y_k = data["y_k"].cpu().numpy()

plt.figure(figsize=(7, 5))
plt.plot(x_val, u_val, "r-", label="Exact")
# plt.plot(x_val,pred_list_u.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val, pred_list_u.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
plt.fill_between(
    x_val.reshape(-1),
    pred_list_u.mean(0).squeeze().T - 2 * pred_list_u.std(0).squeeze().T,
    pred_list_u.mean(0).squeeze().T + 2 * pred_list_u.std(0).squeeze().T,
    facecolor="b",
    alpha=0.2,
    label="2 std",
)
plt.plot(x_u, y_u, "kx", markersize=5, label="Training data")
plt.xlim([lb, ub])
plt.legend(fontsize=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(x_val, k_val, "r-", label="Exact")
# plt.plot(x_val,pred_list_k.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val, pred_list_k.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
plt.fill_between(
    x_val.reshape(-1),
    pred_list_k.mean(0).squeeze().T - 2 * pred_list_k.std(0).squeeze().T,
    pred_list_k.mean(0).squeeze().T + 2 * pred_list_k.std(0).squeeze().T,
    facecolor="b",
    alpha=0.2,
    label="2 std",
)
plt.plot(x_k, y_k, "kx", markersize=5, label="Training data")
plt.xlim([lb, ub])
plt.legend(fontsize=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(x_val, f_val, "r-", label="Exact")
# plt.plot(x_val,pred_list_f.squeeze(2).T, 'b-',alpha=0.01)
plt.plot(x_val, pred_list_f.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
plt.fill_between(
    x_val.reshape(-1),
    pred_list_f.mean(0).squeeze().T - 2 * pred_list_f.std(0).squeeze().T,
    pred_list_f.mean(0).squeeze().T + 2 * pred_list_f.std(0).squeeze().T,
    facecolor="b",
    alpha=0.2,
    label="2 std",
)
plt.plot(x_f, y_f, "kx", markersize=5, label="Training data")
plt.xlim([lb, ub])
plt.legend(fontsize=10)
plt.show()
# %%
