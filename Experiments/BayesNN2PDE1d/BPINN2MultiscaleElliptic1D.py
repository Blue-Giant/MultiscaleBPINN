import os
import sys
import time
import datetime
import platform
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Network import hamiltorch
from Network import BayesNN_Utils
from Network import DNN2Bayes
from Network import BayesDNN
from Problems import Eqs2BayesNN
from utilizers import plot2Bayes
from utilizers import Log_Print2Bayes
from utilizers import saveData
from utilizers import save_load_NetModule
from utilizers import DNN_tools


def print_log_mse_rel(mse2solu=0.01, rel2solu=0.01, mse2force=0.01, rel2force=0.01, log_out=None):
    # 将运行结果打印出来
    print('mean square error of mean-predict and exact for solution: %.10f\n' % mse2solu)
    print('relative error of mean-predict and exact for solution: %.10f\n' % rel2solu)

    print('mean square error of mean-predict and exact for force: %.10f\n' % mse2force)
    print('relative error of mean-predict and exact for force: %.10f\n' % rel2force)

    DNN_tools.log_string('mean square error of mean-predict and exact for solution: %.10f' % mse2solu, log_out)
    DNN_tools.log_string('relative error of mean-predict and exact for solution: %.10f\n\n' % rel2solu, log_out)

    DNN_tools.log_string('mean square error of mean-predict and exact for force: %.10f' % mse2force, log_out)
    DNN_tools.log_string('relative error of mean-predict and exact for force: %.10f\n\n' % rel2force, log_out)


def model_loss_0p5(data, fmodel, params_unflattened, tau_likes, gradients, params_single):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)

    x_f = data["x_f"]
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]

    Aeps = 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5))

    temp1 = torch.sin(2.0 * torch.pi * x_f / 0.5) * (2.0 * torch.pi / 0.5)
    temp2 = (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5)) * (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.5))
    dAeps_dx = temp1/temp2

    pred_f = -1.0*(Aeps*u_xx + dAeps_dx*u_x)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, Aeps, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


def model_loss_0p1(data, fmodel, params_unflattened, tau_likes, gradients, params_single):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)

    x_f = data["x_f"]
    x_f = x_f.detach().requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    u_x = gradients(u, x_f)[0]
    u_xx = gradients(u_x, x_f)[0]

    Aeps = 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1))

    temp1 = torch.sin(2.0 * torch.pi * x_f / 0.1) * (2.0 * torch.pi / 0.1)
    temp2 = (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1)) * (2.0 + torch.cos(2.0 * torch.pi * x_f / 0.1))
    dAeps_dx = temp1/temp2

    pred_f = -1.0*(Aeps*u_xx + dAeps_dx*u_x)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, u_x, u_xx, Aeps, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


def solve_bayes(Rdic=None):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['act_name2Hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    Log_Print2Bayes.dictionary_out2file(R, log_fileout)

    # using PINN mode to update the parameters of DNN or not
    if Rdic['with_gpu'] is True:
        assert (torch.cuda.is_available() is True)
        print(f"Is CUDA available?: {torch.cuda.is_available()}")
        device = 'cuda:0'
    else:
        device = "cpu"

    prior_std = 1
    like_std = Rdic['noise_level']

    lr = Rdic['learning_rate']  # the learning rate for optimizer in PINN model or Hamilton  model
    step2update_lr = Rdic['step2update_lr']
    gamma2update_lr = Rdic['gamma2update_lr']
    open_update_lr = Rdic['update_lr']

    pde = True
    max_epoch = R['max_epoch']  # the total iteration epoch of training for PINN
    burn = 100  # Hamilton认为前burn=100次估计的不准。取后面的B=600-100=500次作为结果
    num_samples = R['sample_num2hamilton']  # the number of samplings for hamilton sampler,Hamilton 抽样次数
    # B= num_samples - burn 为总的抽样次数，即BxNxD, 其中B为总的抽样次数，N为测试点个数，D为解的维度
    L = 100

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    lb = 0.0        # the left boundary of interested interval
    ub = 1.0        # the right boundary of interested interval
    N_tr_u = 50   # the number of sampled points for dealing with solution
    N_tr_f = 50   # the number of sampled points for dealing with force-side
    N_val = 1000  # the number of sampled points for obtaining real solution, parameter and force-side

    # eps = 0.5
    eps = 0.1
    u = lambda x: x-torch.mul(x, x)+(0.25*eps/torch.pi)*torch.sin(torch.pi*2.0*x/eps) - \
                  (0.5*eps/torch.pi)*x*torch.sin(torch.pi*2.0*x/eps)-eps*eps/(4.0*torch.pi**2)*torch.cos(torch.pi*2.0*x/eps)+\
                  0.25*eps/(torch.pi ** 2)
    f = lambda x: torch.ones_like(x)
    k = lambda x: 1.0 / (2.0 + torch.cos(2.0 * torch.pi * x / eps))

    data = {}
    if 'equidistance' == str.lower(Rdic['opt2sampling']):
        xu = np.reshape(np.linspace(lb, ub, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xu)
        data["x_u"] = torch.from_numpy(xu)
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias
        xf = np.reshape(np.linspace(lb, ub, N_tr_f, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xf)
        data["x_f"] = torch.from_numpy(xf)                                       # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias
    elif 'lhs' == str.lower(Rdic['opt2sampling']):
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)
        # torch.linspace(start, end, steps) view making it a column vector, boundary points
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb                     # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias
    else:
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)
        # torch.linspace(start, end, steps) view making it a column vector, boundary points
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb                     # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias

    # exact value of solution, parameter and force-side
    data_val = {}
    data_val["x_u"] = torch.linspace(lb, ub, N_val).view(-1, 1)
    data_val["y_u"] = u(data_val["x_u"])
    data_val["x_f"] = torch.linspace(lb, ub, N_val).view(-1, 1)
    data_val["y_f"] = f(data_val["x_f"])

    for d in data:
        data[d] = data[d].to(device)
    for d in data_val:
        data_val[d] = data_val[d].to(device)

    if 'NET_2HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=5.0,
                                         trainable2ff=Rdic['trainable2ff_layer'],
                                         type2float='float32', to_gpu=False, gpu_no=0).to(device)
    elif 'NET_2HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_2HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_3HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=5.0,
                                         trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                         gpu_no=0).to(device)
    elif 'NET_3HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_3HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_2HIDDEN_MULTISCALE_FOURIER' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_2Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Two_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=True, freq=scale2u).to(device)
    elif 'NET_3HIDDEN_MULTISCALE_FOURIER' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_3Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Three_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=True, freq=scale2u).to(device)
    elif 'NET_2HIDDEN_FOURIER_SUB' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # scale2u = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_2Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u, num2subnets=len(scale2u)).to(device)
    elif 'NET_3HIDDEN_FOURIER_SUB' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_3Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u, num2subnets=len(scale2u)).to(device)
    elif 'NET_2HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_2Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
    elif 'NET_3HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_3Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
    elif 'NET_4HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        net_u = DNN2Bayes.Net_4Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Four_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
    elif 'NET_2HIDDEN' == str.upper(Rdic['model']):
        net_u = BayesDNN.Net_2Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Two_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
    elif 'NET_3HIDDEN' == str.upper(Rdic['model']):
        net_u = BayesDNN.Net_3Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Three_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
    elif 'NET_2HIDDEN_FOURIER' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Two_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
    elif 'NET_3HIDDEN_FOURIER' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Three_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
    else:
        net_u = BayesDNN.Net_4Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Four_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)

    nets = [net_u]

    if 'PINN' == Rdic['mode2update_para']:
        if eps == 0.5:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p5, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
        elif eps == 0.1:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p1, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
        else:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_0p1, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr, tau_priors=tau_priors,
                tau_likes=tau_likes, device=device, pde=pde, total_epochs=max_epoch)
    else:
        if eps == 0.5:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p5, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)
        elif eps == 0.1:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p1, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)
        else:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_0p1, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

    if eps == 0.5:
        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p5, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)
    elif eps == 0.1:
        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p1, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)
    else:
        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, params_hmc, data_val, model_loss=model_loss_0p1, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    Log_Print2Bayes.print_log_validation(Expected, log_out=log_fileout)

    pred_list_u = pred_list[0].cpu().numpy()
    pred_list_f = pred_list[1].cpu().numpy()

    mean2pred_u = np.reshape(pred_list_u.mean(0).squeeze().T, newshape=[-1, 1])
    mean2pred_f = np.reshape(pred_list_f.mean(0).squeeze().T, newshape=[-1, 1])

    if Rdic['with_gpu'] is True:
        x_val = data_val["x_u"].cpu().detach().numpy()
        u_val = data_val["y_u"].cpu().detach().numpy()
        f_val = data_val["y_f"].cpu().detach().numpy()

        x_u = data["x_u"].cpu().detach().numpy()
        y_u = data["y_u"].cpu().detach().numpy()
        x_f = data["x_f"].cpu().detach().numpy()
        y_f = data["y_f"].cpu().detach().numpy()
    else:
        # plot
        x_val = data_val["x_u"].detach().numpy()
        u_val = data_val["y_u"].detach().numpy()
        f_val = data_val["y_f"].detach().numpy()

        x_u = data["x_u"].detach().numpy()
        y_u = data["y_u"].detach().numpy()
        x_f = data["x_f"].detach().numpy()
        y_f = data["y_f"].detach().numpy()

    diff2mean_pred_U = mean2pred_u - u_val
    diff2mean_pred_F = mean2pred_f - f_val

    mse2U = np.mean(np.square(diff2mean_pred_U))
    rel2U = np.sqrt(mse2U/np.mean(np.square(u_val)))

    mse2F = np.mean(np.square(diff2mean_pred_F))
    rel2F = np.sqrt(mse2F / np.mean(np.square(f_val)))

    print_log_mse_rel(mse2solu=mse2U, rel2solu=rel2U, mse2force=mse2F, rel2force=rel2F, log_out=log_fileout)

    saveData.saveTestPoints_Solus2mat(x_val, u_val, mean2pred_u, name2point_data='x',
                                      name2solu_exact='Uexact_'+str(R['act_name2Hidden']),
                                      name2solu_predict='Umean_'+str(R['act_name2Hidden']),
                                      file_name='solu2test', outPath=R['FolderName'])

    saveData.saveTestPoints_Solus2mat(x_val, f_val, mean2pred_f, name2point_data='x',
                                      name2solu_exact='Fexact_' + str(R['act_name2Hidden']),
                                      name2solu_predict='Fmean_' + str(R['act_name2Hidden']),
                                      file_name='force2test', outPath=R['FolderName'])

    saveData.saveTrainData2mat(x_u, y_u, name2point_data='xu_train',
                               name2solu_exact='Utrain_' + str(R['act_name2Hidden']),
                               file_name='solu2train', outPath=R['FolderName'])

    saveData.saveTrainData2mat(x_f, y_f, name2point_data='xf_train',
                               name2solu_exact='Ftrain_' + str(R['act_name2Hidden']),
                               file_name='force2train', outPath=R['FolderName'])

    plot2Bayes.plot2u(x_val=x_val, u_val=u_val, pred_list_u=pred_list_u, x_u=x_u, y_u=y_u, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='solu2u')

    plot2Bayes.plot2f(x_val=x_val, f_val=f_val, pred_list_f=pred_list_f, x_f=x_f, y_f=y_f, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='force')

    save_load_NetModule.save_bayes_net2file_with_keys(
        outPath=R['FolderName'], model2net=net_u, paras2net=params_hmc, name2model='Solu', learning_rate=lr,
        expected_log_prob=Expected, epoch=R['max_epoch'], opt2update_para=Rdic['mode2update_para'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    # 默认使用 GPU，这个标记就不要设为-1，设为0,1,2,3,4....n（n指GPU的数目，即电脑有多少块GPU）
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # -1代表使用 CPU 模式
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的GPU设备仅为第 0 块GPU, 设备名称为'/gpu:0'
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

    # ------------------------------------------- 文件保存路径设置 ----------------------------------------
    file2results = 'Results'
    store_file = 'BPINN2MultiScaleElliptic1D_Test'
    BASE_DIR2FILE = os.path.dirname(os.path.abspath(__file__))
    split_BASE_DIR2FILE = os.path.split(BASE_DIR2FILE)
    split_BASE_DIR2FILE = os.path.split(split_BASE_DIR2FILE[0])
    BASE_DIR = split_BASE_DIR2FILE[0]
    sys.path.append(BASE_DIR)
    OUT_DIR_BASE = os.path.join(BASE_DIR, file2results)
    OUT_DIR = os.path.join(OUT_DIR_BASE, store_file)
    sys.path.append(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    current_day_time = datetime.datetime.now()                 # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')

    # The setups of Problems
    R['PDE_type'] = 'MultiScale'
    R['equa_name'] = 'MultiScale_Elliptic'

    # The setups of DNN for approximating solution, parameter and force-side
    # R['model'] = 'Net_2Hidden'
    R['model'] = 'Net_2Hidden_FF'
    # R['model'] = 'Net_2Hidden_2FF'
    # R['model'] = 'Net_2Hidden_3FF'
    # R['model'] = 'Net_2Hidden_Fourier'
    # R['model'] = 'Net_2Hidden_Fourier_sub'
    # R['model'] = 'Net_2Hidden_Multiscale'
    # R['model'] = 'Net_2Hidden_Multiscale_Fourier'

    # R['model'] = 'Net_3Hidden'
    # R['model'] = 'Net_3Hidden_FF'
    # R['model'] = 'Net_3Hidden_2FF'
    # R['model'] = 'Net_3Hidden_3FF'
    # R['model'] = 'Net_3Hidden_Fourier'
    # R['model'] = 'Net_3Hidden_Fourier_sub'
    # R['model'] = 'Net_3Hidden_Multiscale'
    # R['model'] = 'Net_3Hidden_Multiscale_Fourier'

    # R['model'] = 'Net_4Hidden'
    # R['model'] = 'Net_4Hidden_Multiscale'

    OUT_DIR_PDE = os.path.join(OUT_DIR, str(R['equa_name']))  # 路径连
    sys.path.append(OUT_DIR_PDE)
    if not os.path.exists(OUT_DIR_PDE):
        print('---------------------- OUT_DIR_PDE ---------------------:', OUT_DIR_PDE)
        os.mkdir(OUT_DIR_PDE)

    Module_Time = str(R['model']) + '_' + str(date_time_dir)
    FolderName = os.path.join(OUT_DIR_PDE, Module_Time)          # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)
    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    R['indim'] = 1
    R['outdim'] = 1

    # R['opt2sampling'] = 'equidistance'
    R['opt2sampling'] = 'lhs'
    # R['opt2sampling'] = 'random'

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden':
        R['Two_hidden_layer'] = [30, 30]
        R['Three_hidden_layer'] = [30, 30, 30]
        R['Four_hidden_layer'] = [30, 30, 30, 30]
    if R['model'] == 'Net_2Hidden_Fourier' or R['model'] == 'Net_3Hidden_Fourier':
        R['Two_hidden_layer'] = [15, 30]
        R['Three_hidden_layer'] = [15, 30, 30]
        R['Four_hidden_layer'] = [15, 30, 30, 30]
    elif R['model'] == 'Net_2Hidden_Fourier_sub' or R['model'] == 'Net_3Hidden_Fourier_sub':
        R['Two_hidden_layer'] = [5, 10]
        R['Three_hidden_layer'] = [5, 10, 10]
        R['Four_hidden_layer'] = [5, 10, 10, 10]
    else:
        R['Two_hidden_layer'] = [30, 30]
        R['Three_hidden_layer'] = [30, 30, 30]
        R['Four_hidden_layer'] = [30, 30, 30, 30]

    # R['Two_hidden_layer'] = [15, 40]
    # R['Three_hidden_layer'] = [15, 40, 20]

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden' or R['model'] == 'Net_4Hidden':
        # R['act_name2Input'] = 'tanh'
        # R['act_name2Input'] = 'enh_tanh'
        R['act_name2Input'] = 'sin'
        # R['act_name2Input'] = 'gelu'
        # R['act_name2Input'] = 'sinAddcos'
    else:
        R['act_name2Input'] = 'fourier'

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden' or R['model'] == 'Net_4Hidden':
        # R['act_name2Hidden'] = 'tanh'
        R['act_name2Hidden'] = 'sin'
    else:
        # R['act_name2Hidden'] = 'enh_tanh'
        R['act_name2Hidden'] = 'sin'
        # R['act_name2Hidden'] = 'silu'
        # R['act_name2Hidden'] = 'gelu'
        # R['act_name2Hidden'] = 'sinAddcos'

    R['act_name2Output'] = 'linear'

    # R['mode2update_para'] = 'PINN'
    R['mode2update_para'] = 'Hamilton'

    # R['noise_level'] = 0.01
    R['noise_level'] = 0.05
    # R['noise_level'] = 0.1

    R['trainable2ff_layer'] = True

    R['with_gpu'] = True

    R['initWB'] = True

    if R['mode2update_para'] == 'PINN':
        R['update_lr'] = False
        # R['learning_rate'] = 0.01     # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0025   # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.0005  # this is the learning rate for optimizer in PINN  model
        R['learning_rate'] = 0.0001  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00005  # this is the learning rate for optimizer in PINN  model
        # R['learning_rate'] = 0.00001  # this is the learning rate for optimizer in PINN  model
        R['step2update_lr'] = 100
        R['gamma2update_lr'] = 0.97

        R['sample_num2hamilton'] = 600
        stop_flag2pinn = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
        R['activate_stop2pinn'] = int(stop_flag2pinn)
        # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
        R['max_epoch'] = 200000
        if 0 != R['activate_stop2pinn']:
            max_epoch2training = input('please input a stop epoch:')
            R['max_epoch'] = int(max_epoch2training)
    else:
        R['update_lr'] = False
        # R['learning_rate'] = 0.0025   # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.005  # this is the learning rate for optimizer in Hamilton  model
        R['learning_rate'] = 0.001  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.0005  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.0001  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.00005  # this is the learning rate for optimizer in Hamilton  model
        # R['learning_rate'] = 0.00001  # this is the learning rate for optimizer in Hamilton  model
        R['step2update_lr'] = 25
        R['gamma2update_lr'] = 0.85

        R['sample_num2hamilton'] = 600
        R['max_epoch'] = 20000

    solve_bayes(Rdic=R)
    # 隐藏层激活函数：sin 比 tanh 好, 也比sin+cos 好, 也比 gelu 好, 也比 enh_tanh 好
    # 采样方法：均匀采样，然后打乱，比随机采样好

    # Under the case of learning rate: 0.0005, the performance of net_2hidden and net_3hidden are good for smooth

    # Under the case of learning rate: 0.0001
    # For smooth, the performance of net_2hidden and net_3hidden with hidden units [20,20] and [20, 20, 20]
    # will become degraded.
    # The rel of solution for net_2hidden is 0.6248024106   The rel of solution for net_3hidden is 0.6238239408
    #     Net2Hidden_FF perform best for PDE1
    #     Net2Hidden_FF perform best for PDE2
    #     Net2Hidden_FF perform best for PDE3
    #     Net2Hidden_2FF perform best for PDE4

    # Net2Hidden_FourierBaisi, for PDE3, sin 作为激活函数，效果最好。比 tanh 好, 也比sin+cos 好, 也比 gelu 好, 也比 enh_tanh 好
    # Net2Hidden_FourierBaisi, for PDE1, sin 作为激活函数，效果比 全sin激活函数的Net2Hidden好。全tanh激活函数的Net2Hidden效果很差

    # 对于光滑的函数，使用FourierFeature网络时，如2FF,sigma要小，反之sigma要大些

