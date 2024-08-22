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


def calculate_snr(signal=None, noise=None):
    power2signal = torch.mean(torch.square(signal), dim=0)
    power2noise = torch.mean(torch.square(noise), dim=0)
    snr = 10.0*torch.log10(power2signal/power2noise)
    return snr


def print_log_SNR(snr2solu=0.01, snr2kcoef=0.01, snr2fside=0.01, log_out=None):
    # 将运行结果打印出来
    print('SNR of solution for training data: %.10f' % snr2solu)
    print('SNR of k-coeff for training data: %.10f' % snr2kcoef)
    print('SNR of force-side for training data: %.10f\n' % snr2fside)

    DNN_tools.log_string('SNR of solution for training data: %.10f' % snr2solu, log_out)
    DNN_tools.log_string('SNR of k-coeff for training data: %.10f' % snr2kcoef, log_out)
    DNN_tools.log_string('SNR of force-side for training data: %.10f\n\n' % snr2fside, log_out)


def log_print_run_time(run_time=0.01, log_fileout=None):
    print('running time:', run_time)
    DNN_tools.log_string('The running time: %.18f s\n' % run_time, log_fileout)


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


def solve_bayes(Rdic=None):
    log_out_path = Rdic['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', Rdic['act_name2Hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    Log_Print2Bayes.dictionary_out2file(Rdic, log_fileout)

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
    max_epoch = Rdic['max_epoch']  # the total iteration epoch of training for PINN
    burn = 100  # Hamilton认为前burn=100次估计的不准。取后面的B=600-100=500次作为结果
    num_samples = Rdic['sample_num2hamilton']  # the number of samplings for hamilton sampler,Hamilton 抽样次数
    # B= num_samples - burn 为总的抽样次数，即BxNxD, 其中B为总的抽样次数，N为测试点个数，D为解的维度
    L = 100

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    lb = 0        # the left boundary of interested interval
    ub = 1        # the right boundary of interested interval
    N_tr_u = 50   # the number of sampled points for dealing with solution
    N_tr_f = 50   # the number of sampled points for dealing with force-side
    N_tr_k = 25   # the number of sampled points for dealing with parameter
    N_val = 1000  # the number of sampled points for obtaining real solution, parameter and force-side

    u, k, f = Eqs2BayesNN.get_infos_1d(equa_name=Rdic['equa_name'])   # get the infos for PDE problem

    data = {}
    if 'equidistance' == str.lower(Rdic['opt2sampling']):
        xu = np.reshape(np.linspace(lb, ub, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xu)
        data["x_u"] = torch.from_numpy(xu)              # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                        # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std  # the noisy data
        data["y_u"] = clean_u + noise2u                 # adding bias

        xf = np.reshape(np.linspace(lb, ub, N_tr_f, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xf)
        data["x_f"] = torch.from_numpy(xf)              # interior points for governed equation
        clean_f = f(data["x_f"])                        # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std  # the noisy data
        data["y_f"] = clean_f + noise2f                 # adding bias

        xk = np.reshape(np.linspace(lb, ub, N_tr_k, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xk)
        data["x_k"] = torch.from_numpy(xk)              # interior points
        clean_k = k(data["x_k"])                        # the exact coefficient without noise on interior points
        noise2k = torch.randn_like(clean_k) * like_std  # the noisy data
        data["y_k"] = clean_k + noise2k                 # adding bias
    elif 'lhs' == str.lower(Rdic['opt2sampling']):
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)  # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                        # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std  # the noisy data
        data["y_u"] = clean_u + noise2u                 # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb   # interior points
        clean_f = f(data["x_f"])                               # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std         # the noisy data
        data["y_f"] = clean_f + noise2f                        # adding bias

        data["x_k"] = (ub - lb) * torch.rand(N_tr_k, 1) + lb   # interior points
        clean_k = k(data["x_k"])                               # the exact coefficient without noise on interior points
        noise2k = torch.randn_like(clean_k) * like_std         # the noisy data
        data["y_k"] = clean_k + noise2k                        # adding bias
    else:
        temp1 = torch.linspace(lb, ub, 2).view(-1, 1)
        temp2 = (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb
        data["x_u"] = torch.cat((temp1, temp2), dim=0)        # interior points for solution including Dirichlet boundary
        clean_u = u(data["x_u"])                              # the exact solution without noise on interior points
        noise2u = torch.randn_like(clean_u) * like_std        # the noisy data
        data["y_u"] = clean_u + noise2u                       # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb  # interior points
        clean_f = f(data["x_f"])                              # the exact force-side without noise on interior points
        noise2f = torch.randn_like(clean_f) * like_std        # the noisy data
        data["y_f"] = clean_f + noise2f                       # adding bias

        data["x_k"] = (ub - lb) * torch.rand(N_tr_k, 1) + lb  # interior points
        clean_k = k(data["x_k"])                              # the exact coefficient without noise on interior points
        noise2k = torch.randn_like(clean_k) * like_std        # the noisy data
        data["y_k"] = clean_k + noise2k                       # adding bias

    snr2solu = calculate_snr(signal=clean_u, noise=noise2u)
    snr2coef = calculate_snr(signal=clean_k, noise=noise2k)
    snr2force_side = calculate_snr(signal=clean_f, noise=noise2f)
    print_log_SNR(snr2solu=snr2solu, snr2kcoef=snr2coef, snr2fside=snr2force_side, log_out=log_fileout)

    # exact value of solution, parameter and force-side
    data_val = {}
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

    if 'NET_2HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=10.0,
                                         trainable2ff=Rdic['trainable2ff_layer'],
                                         type2float='float32', to_gpu=False, gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=2.0,
                                         trainable2ff=Rdic['trainable2ff_layer'],
                                         type2float='float32', to_gpu=False, gpu_no=0).to(device)
    elif 'NET_2HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=2.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_2HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=2.5, sigma3=5.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_3HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=5.0,
                                         trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                         gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'],
                                         actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                         sigma=2.5, trainable2ff=Rdic['trainable2ff_layer'], type2float='float32',
                                         to_gpu=False, gpu_no=0).to(device)
    elif 'NET_3HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_3HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=2.5, sigma3=5,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_2HIDDEN_MULTISCALE_FOURIER' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scale2k = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_2Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Two_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_2Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Two_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=False, freq=scale2k).to(device)
    elif 'NET_3HIDDEN_MULTISCALE_FOURIER' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scale2k = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_3Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Three_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Three_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                  type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                  repeat_Highfreq=False, freq=scale2k).to(device)
    elif 'NET_2HIDDEN_FOURIER_SUB' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # scale2u = np.array([1, 2, 3, 4, 5])
        scale2k = np.array([1, 2, 3])
        net_u = DNN2Bayes.Net_2Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u, num2subnets=len(scale2u)).to(device)
        net_k = DNN2Bayes.Net_2Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=False, freq=scale2k, num2subnets=len(scale2k)).to(device)
    elif 'NET_3HIDDEN_FOURIER_SUB' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # scale2u = np.array([1, 2, 3, 4, 5])
        scale2k = np.array([1, 2, 3])
        net_u = DNN2Bayes.Net_3Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u, num2subnets=len(scale2u)).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FourierSub(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=False, freq=scale2k, num2subnets=len(scale2k)).to(device)
    elif 'NET_2HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scale2k = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_2Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_2Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Two_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=False, freq=scale2k).to(device)
    elif 'NET_3HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scale2k = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_3Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_3Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Three_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=False, freq=scale2k).to(device)
    elif 'NET_4HIDDEN_MULTISCALE' == str.upper(Rdic['model']):
        scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        scale2k = np.array([1, 2, 3, 4, 5])
        net_u = DNN2Bayes.Net_4Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Four_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_4Hidden_MultiScale(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                 hidden_layer=Rdic['Four_hidden_layer'],
                                                 actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                 type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0,
                                                 repeat_Highfreq=False, freq=scale2k).to(device)
    elif 'NET_2HIDDEN' == str.upper(Rdic['model']):
        net_u = BayesDNN.Net_2Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Two_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
        net_k = BayesDNN.Net_2Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Two_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
    elif 'NET_3HIDDEN' == str.upper(Rdic['model']):
        net_u = BayesDNN.Net_3Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Three_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
        net_k = BayesDNN.Net_3Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Three_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
    elif 'NET_2HIDDEN_FOURIER' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Two_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Two_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
    elif 'NET_3HIDDEN_FOURIER' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Three_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FourierBasis(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                   hidden_layer=Rdic['Three_hidden_layer'],
                                                   actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                                   type2float='float32', to_gpu=Rdic['with_gpu'], gpu_no=0).to(device)
    else:
        net_u = BayesDNN.Net_4Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Four_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)
        net_k = BayesDNN.Net_4Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'], hidden_layer=Rdic['Four_hidden_layer'],
                                     actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                     type2float='float32', to_gpu=False, gpu_no=0, init_W_B=Rdic['initWB']).to(device)

    nets = [net_u, net_k]

    # # sampling!! The training data is fixed for all training process? why? Can it be varied? No, it is fixed
    time_begin = time.time()
    if 'PINN' == Rdic['mode2update_para']:
        params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
            nets, data, model_loss=model_loss, learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
            gamma2change_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
            total_epochs=max_epoch)
    else:
        params_hmc = BayesNN_Utils.update_paras_by_hamilton(
            nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L,
            learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
            burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

    pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
        nets, params_hmc, data_val, model_loss=model_loss, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

    time_end = time.time()
    run_time = time_end - time_begin

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    Log_Print2Bayes.print_log_validation(Expected, log_out=log_fileout)

    pred_list_u = pred_list[0].cpu().numpy()
    pred_list_k = pred_list[1].cpu().numpy()
    pred_list_f = pred_list[2].cpu().numpy()

    mean2pred_u = np.reshape(pred_list_u.mean(0).squeeze().T, newshape=[-1, 1])
    mean2pred_k = np.reshape(pred_list_k.mean(0).squeeze().T, newshape=[-1, 1])
    mean2pred_f = np.reshape(pred_list_f.mean(0).squeeze().T, newshape=[-1, 1])

    if Rdic['with_gpu'] is True:
        x_val = data_val["x_u"].cpu().detach().numpy()
        u_val = data_val["y_u"].cpu().detach().numpy()
        k_val = data_val["y_k"].cpu().detach().numpy()
        f_val = data_val["y_f"].cpu().detach().numpy()

        x_u = data["x_u"].cpu().detach().numpy()
        y_u = data["y_u"].cpu().detach().numpy()
        x_f = data["x_f"].cpu().detach().numpy()
        y_f = data["y_f"].cpu().detach().numpy()
        x_k = data["x_k"].cpu().detach().numpy()
        y_k = data["y_k"].cpu().detach().numpy()
    else:
        # plot
        x_val = data_val["x_u"].detach().numpy()
        u_val = data_val["y_u"].detach().numpy()
        k_val = data_val["y_k"].detach().numpy()
        f_val = data_val["y_f"].detach().numpy()

        x_u = data["x_u"].detach().numpy()
        y_u = data["y_u"].detach().numpy()
        x_f = data["x_f"].detach().numpy()
        y_f = data["y_f"].detach().numpy()
        x_k = data["x_k"].detach().numpy()
        y_k = data["y_k"].detach().numpy()

    diff2mean_pred_U = mean2pred_u - u_val
    diff2mean_pred_K = mean2pred_k - k_val
    diff2mean_pred_F = mean2pred_f - f_val

    mse2U = np.mean(np.square(diff2mean_pred_U))
    rel2U = np.sqrt(mse2U/np.mean(np.square(u_val)))

    mse2K = np.mean(np.square(diff2mean_pred_K))
    rel2K = np.sqrt(mse2K / np.mean(np.square(k_val)))

    mse2F = np.mean(np.square(diff2mean_pred_F))
    rel2F = np.sqrt(mse2F / np.mean(np.square(k_val)))

    Log_Print2Bayes.print_log_mse_rel(mse2solu=mse2U, rel2solu=rel2U, mse2para=mse2K, rel2para=rel2K, mse2force=mse2F,
                                      rel2force=rel2F, log_out=log_fileout)

    saveData.saveTestPoints_Solus2mat(x_val, u_val, mean2pred_u, name2point_data='x',
                                      name2solu_exact='Uexact_'+str(Rdic['act_name2Hidden']),
                                      name2solu_predict='Umean_'+str(Rdic['act_name2Hidden']),
                                      file_name='solu2test', outPath=Rdic['FolderName'])

    saveData.saveTestPoints_Solus2mat(x_val, k_val, mean2pred_k, name2point_data='x',
                                      name2solu_exact='Kexact_' + str(Rdic['act_name2Hidden']),
                                      name2solu_predict='Kmean_' + str(Rdic['act_name2Hidden']),
                                      file_name='para2test', outPath=Rdic['FolderName'])

    saveData.saveTestPoints_Solus2mat(x_val, f_val, mean2pred_f, name2point_data='x',
                                      name2solu_exact='Fexact_' + str(Rdic['act_name2Hidden']),
                                      name2solu_predict='Fmean_' + str(Rdic['act_name2Hidden']),
                                      file_name='force2test', outPath=Rdic['FolderName'])

    saveData.saveTrainData2mat(x_u, y_u, name2point_data='xu_train',
                               name2solu_exact='Utrain_' + str(Rdic['act_name2Hidden']),
                               file_name='solu2train', outPath=Rdic['FolderName'])

    saveData.saveTrainData2mat(x_k, y_k, name2point_data='xk_train',
                               name2solu_exact='Ktrain_' + str(Rdic['act_name2Hidden']),
                               file_name='para2train', outPath=Rdic['FolderName'])

    saveData.saveTrainData2mat(x_f, y_f, name2point_data='xf_train',
                               name2solu_exact='Ftrain_' + str(Rdic['act_name2Hidden']),
                               file_name='force2train', outPath=Rdic['FolderName'])

    plot2Bayes.plot2u(x_val=x_val, u_val=u_val, pred_list_u=pred_list_u, x_u=x_u, y_u=y_u, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='solu2u')

    plot2Bayes.plot2k(x_val=x_val, k_val=k_val, pred_list_k=pred_list_k, x_k=x_k, y_k=y_k, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='para2k')

    plot2Bayes.plot2f(x_val=x_val, f_val=f_val, pred_list_f=pred_list_f, x_f=x_f, y_f=y_f, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='force')

    save_load_NetModule.save_bayes_net2file_with_keys(
        outPath=Rdic['FolderName'], model2net=net_u, paras2net=params_hmc, name2model='Solu', learning_rate=lr,
        expected_log_prob=Expected, epoch=Rdic['max_epoch'], opt2update_para=Rdic['mode2update_para'])
    save_load_NetModule.save_bayes_net2file_with_keys(
        outPath=Rdic['FolderName'], model2net=net_k, paras2net=params_hmc, name2model='Para', learning_rate=lr,
        expected_log_prob=Expected, epoch=Rdic['max_epoch'], opt2update_para=Rdic['mode2update_para'])

    log_print_run_time(run_time=run_time, log_fileout=log_fileout)


def LOOP_MBPINN(learning_rate=0.001, noise_level=0.01, model2NN='Net_2Hidden',
                Method2UpdatePara='PINN', act_name2hidden='sin', opt2sampling='random'):
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
    store_file = 'Bayes1D'
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

    # The setups of Problems
    R['PDE_type'] = 'NoninearPossion'
    # R['equa_name'] = 'PDE1'
    R['equa_name'] = 'PDE2'     # The solution have multi frequency component  sin(2*pi*x)+0.1*cos(10*pi*x)
    # R['equa_name'] = 'PDE3'     # The solution have multi frequency component  sin(5*pi*x)+0.1*sin(10*pi*x)
    # R['equa_name'] = 'PDE4'       # The solution have multi frequency component  sin(5*pi*x)+0.5*sin(10*pi*x)
    # R['equa_name'] = 'PDE5'     # The solution have multi frequency component  sin(2*pi*x)+0.2*sin(10*pi*x)

    # The setups of DNN for approximating solution, parameter and force-side
    # R['model'] = model2NN
    R['model'] = model2NN
    # R['model'] = model2NN
    PDE_EQUA = R['PDE_type'] + '_' + R['equa_name']

    OUT_DIR_PDE = os.path.join(OUT_DIR, str(PDE_EQUA))  # 路径连
    sys.path.append(OUT_DIR_PDE)
    if not os.path.exists(OUT_DIR_PDE):
        print('---------------------- OUT_DIR_PDE ---------------------:', OUT_DIR_PDE)
        os.mkdir(OUT_DIR_PDE)

    Module_Time = str(R['model']) + '_' + str(Method2UpdatePara) + '_' + str(act_name2hidden) + '_Lr' + str(learning_rate)+ '_Noise' + str(noise_level)
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

    R['opt2sampling'] = opt2sampling
    # R['opt2sampling'] = 'random'

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden':
        R['Two_hidden_layer'] = [30, 30]
        R['Three_hidden_layer'] = [30, 30, 30]
        R['Four_hidden_layer'] = [30, 30, 30, 30]
    if R['model'] == 'Net_2Hidden_Fourier' or R['model'] == 'Net_3Hidden_Fourier':
        R['Two_hidden_layer'] = [15, 30]
        R['Three_hidden_layer'] = [15, 30, 30]
        R['Four_hidden_layer'] = [15, 30, 30, 30]
    else:
        R['Two_hidden_layer'] = [30, 30]
        R['Three_hidden_layer'] = [30, 30, 30]
        R['Four_hidden_layer'] = [30, 30, 30, 30]

    # R['Two_hidden_layer'] = [15, 40]
    # R['Three_hidden_layer'] = [15, 40, 20]

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden' or R['model'] == 'Net_4Hidden':
        # R['act_name2Input'] = 'tanh'
        # R['act_name2Input'] = 'enh_tanh'
        R['act_name2Input'] = act_name2hidden
        # R['act_name2Input'] = 'gelu'
        # R['act_name2Input'] = 'sinAddcos'
    else:
        R['act_name2Input'] = 'fourier'

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden' or R['model'] == 'Net_4Hidden':
        # R['act_name2Hidden'] = 'tanh'
        R['act_name2Hidden'] = act_name2hidden
    else:
        # R['act_name2Hidden'] = 'enh_tanh'
        R['act_name2Hidden'] = act_name2hidden
        # R['act_name2Hidden'] = 'silu'
        # R['act_name2Hidden'] = 'gelu'
        # R['act_name2Hidden'] = 'sinAddcos'

    R['act_name2Output'] = 'linear'

    # R['mode2update_para'] = Method2UpdatePara
    R['mode2update_para'] = Method2UpdatePara

    R['noise_level'] = noise_level

    R['trainable2ff_layer'] = True

    R['with_gpu'] = True

    R['initWB'] = True

    if str.upper(R['mode2update_para']) == 'PINN':
        R['update_lr'] = False
        R['learning_rate'] = learning_rate     # this is the learning rate for optimizer in PINN  model
        R['step2update_lr'] = 100
        R['gamma2update_lr'] = 0.97

        R['sample_num2hamilton'] = 600
        # stop_flag2pinn = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
        R['activate_stop2pinn'] = 0
        # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
        R['max_epoch'] = 20000
        if 0 != R['activate_stop2pinn']:
            max_epoch2training = input('please input a stop epoch:')
            R['max_epoch'] = int(max_epoch2training)
    else:
        R['update_lr'] = False
        R['learning_rate'] = learning_rate   # this is the learning rate for optimizer in Hamilton  model
        R['step2update_lr'] = 25
        R['gamma2update_lr'] = 0.85

        R['sample_num2hamilton'] = 600
        R['max_epoch'] = 20000

    solve_bayes(Rdic=R)


if __name__ == "__main__":
    # Study the influence of hidden layer, each hidden layer has 30 units
    # NN_Model = 'Net_2Hidden'
    NN_Model = 'Net_2Hidden_FF'
    # NN_Model = 'Net_2Hidden_2FF'

    # NN_Model = 'Net_3Hidden'
    # NN_Model = 'Net_3Hidden_FF'
    # NN_Model = 'Net_3Hidden_2FF'

    update_para = 'PINN'
    noise = 0.05

    LOOP_MBPINN(learning_rate=0.005, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.001, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.0005, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.0001, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.00005, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.00001, noise_level=0.05, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                act_name2hidden='sin', opt2sampling='random')

    LOOP_MBPINN(learning_rate=0.005, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.001, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.0005, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.0001, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.00005, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')
    LOOP_MBPINN(learning_rate=0.00001, noise_level=0.1, model2NN='Net_3Hidden_FF', Method2UpdatePara=update_para,
                opt2sampling='random')


