import torch
from Network import hamiltorch
import matplotlib.pyplot as plt

import torch.nn as nn
import numpy as np
from Network import DNN2Bayes

hamiltorch.set_random_seed(123)
torch.manual_seed(123)
np.random.seed(123)
device = 'cpu'


def generate(num, sigma, lam1, lam2):
    # positions
    lb = -0.7
    rb = 0.7
    X = np.linspace(lb, rb, num)[:, None]
    lb, rb = np.array([[lb]]), np.array([[rb]])
    # values (could be changed if needed)
    y = lam1 * (-1.08)*np.sin(6*X)*(
        np.sin(6*X)**2-2*np.cos(6*X)**2)
    X = np.concatenate([X, lb, rb], axis=0)
    y = np.concatenate([y, np.sin(6*lb) ** 3 * lam2, np.sin(6*rb) ** 3 * lam2], axis=0)
    y = y * (1 + sigma * np.random.randn(*y.shape))
    return X, y


class PoissonPINN(torch.nn.Module):
    def __init__(self, width, lam1, lam2):
        super(PoissonPINN, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        )
        self.lam1 = lam1
        self.lam2 = lam2

    def forward(self, X):
        x = X[:-2].requires_grad_(True)
        u = self.fnn(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_bd = self.fnn(X[-2:])
        return torch.cat([u_xx * self.lam1 * 0.01, u_bd * self.lam2], dim=0)


class PINN2Possion(torch.nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hiddens=None, Model_name='DNN', actName2In='tanh', actName2Hidden='tanh',
                 actName2Out='linear', lam1=1.0, lam2=1.0):
        super(PINN2Possion, self).__init__()
        if 'NET_2HIDDEN_FF' == str.upper(Model_name):
            self.fnn = DNN2Bayes.Net_2Hidden_FF(indim=in_dim, outdim=out_dim, hidden_layer=hiddens,
                                                actName2in=actName2In, actName=actName2Hidden, actName2out=actName2Out,
                                                sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False,
                                                gpu_no=0)
        elif 'NET_2HIDDEN_2FF' == str.upper(Model_name):
            self.fnn = DNN2Bayes.Net_2Hidden_2FF(indim=in_dim, outdim=out_dim, hidden_layer=hiddens,
                                                 actName2in=actName2In, actName=actName2Hidden, actName2out=actName2Out,
                                                 sigma1=1.0, sigma2=8.0, trainable2ff=False, type2float='float32',
                                                 to_gpu=False, gpu_no=0)
        elif 'NET_3HIDDEN_FF' == str.upper(Model_name):
            self.fnn = DNN2Bayes.Net_3Hidden_FF(indim=in_dim, outdim=out_dim, hidden_layer=hiddens,
                                                actName2in=actName2In, actName=actName2Hidden, actName2out=actName2Out,
                                                sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False,
                                                gpu_no=0)
        elif 'NET_3HIDDEN_2FF' == str.upper(Model_name):
            self.fnn = DNN2Bayes.Net_3Hidden_2FF(indim=in_dim, outdim=out_dim, hidden_layer=hiddens,
                                                 actName2in=actName2In, actName=actName2Hidden, actName2out=actName2Out,
                                                 sigma1=1.0, sigma2=5.0, trainable2ff=False, type2float='float32',
                                                 to_gpu=False, gpu_no=0)
        elif 'NET_2HIDDEN' == str.upper(Model_name):
            layer_sizes = np.concatenate([[1], hiddens, [1]], axis=0)
            self.fnn = DNN2Bayes.Net_2Hidden(layer_sizes, activation=torch.tanh, sigma=1.0, trainable2sigma=False,
                                             type2float='float32', to_gpu=False, gpu_no=0)

        elif 'NET_3HIDDEN' == str.upper(Model_name):
            layer_sizes = np.concatenate([[1], hiddens, [1]], axis=0)
            self.fnn = DNN2Bayes.Net_3Hidden(layer_sizes, activation=torch.tanh, sigma=1.0, trainable2sigma=False,
                                             type2float='float32', to_gpu=False, gpu_no=0)

        self.lam1 = lam1
        self.lam2 = lam2

    def forward(self, X):
        x = X[:-2].requires_grad_(True)
        u = self.fnn(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_bd = self.fnn(X[-2:])
        return torch.cat([u_xx * self.lam1 * 0.01, u_bd * self.lam2], dim=0)


width = 50
# sigma = 0.01
sigma = 0.05
lam1 = 1/sigma
lam2 = 1/sigma

# net = PoissonPINN(width, lam1, lam2)
# for param in net.parameters():
#     torch.nn.init.normal_(param)

# layers2hidden = (10, 20)
layers2hidden = (20, 20, 20)
# model = 'Net_2Hidden_FF'
# model = 'Net_2Hidden_2FF'
# model = 'Net_3Hidden_FF'
# model = 'Net_3Hidden_2FF'

# model = 'Net_2Hidden'
model = 'Net_3Hidden'

net = PINN2Possion(in_dim=1, out_dim=1, hiddens=layers2hidden, Model_name=model, actName2In='fourier',
                   actName2Hidden='tanh', actName2Out='linear', lam1=lam1, lam2=lam2)

train_num = 25
test_num = 500

X_train, y_train = generate(train_num, sigma, lam1, lam2)
X_test, y_test = generate(test_num, 0, lam1, lam2)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

tau_list = []
tau = 1.              # /100. # iris 1/10
for w in net.parameters():
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

params_init = hamiltorch.util.flatten(net).to(device).clone()

# May need to tune step_size and L to make sure acceptance rate is around 0.5-0.8
step_size = 0.0005
burn = 2000
num_samples = 5000
L = 20

params_hmc = hamiltorch.sample_model(net, X_train, y_train, model_loss='regression', params_init=params_init,
                                     num_samples=num_samples, step_size=step_size, burn=burn, num_steps_per_sample=L,
                                     tau_list=tau_list, tau_out=1)

y_pred_list = []
for i in range(num_samples - burn):
    params = hamiltorch.util.unflatten(net, params_hmc[i])
    hamiltorch.util.update_model_params_in_place(net, params)
    y_pred = net(X_test)[:-2]
    y_pred_list.append(y_pred)
y_pred = torch.stack(y_pred_list)

# print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))

y_mean = torch.mean(y_pred, dim=0)
y_std = torch.std(y_pred, dim=0)
y_up, y_low = y_mean - 2 * y_std, y_mean + 2 * y_std

plt.plot(X_test[:-2].detach().cpu().numpy(), y_mean.detach().cpu().numpy(), label = 'mean')
plt.plot(X_test[:-2].detach().cpu().numpy(), y_up.detach().cpu().numpy())
plt.plot(X_test[:-2].detach().cpu().numpy(), y_low.detach().cpu().numpy())
plt.plot(X_test[:-2].detach().cpu().numpy(), y_test[:-2].detach().cpu().numpy(), label = 'True')
plt.scatter(X_train[:-2].detach().cpu().numpy(), y_train[:-2].detach().cpu().numpy(), label = 'Data')
plt.legend()
plt.ylabel('u_xx')
plt.xlabel('x')
plt.show()