import os
import sys
import time
import datetime
import platform
import shutil
import math

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

from utilizers import DNN_tools
from utilizers import Log_Print2Bayes
from utilizers import saveData
from utilizers import save_load_NetModule
import scipy.io as scio


def std2list(input_list: list = None, mean2list: float = 0.1, len2list: int = 4):
    diff_list = []
    for i in range(len2list):
        temp = input_list[i]-mean2list
        diff_list.append(temp*temp)
    std = math.sqrt(sum(diff_list)/len2list)
    return std


def print_log_mean_std2solu_mse_rel(mean2mse: float = 0.01, std2mse: float = 0.01, mean2rel: float = 0.01,
                                    std2rel: float = 0.01, log_out=None):
    print('mean value to mse of mean-predict and exact for solution: %.10f' % mean2mse)
    print('std value to mse of mean-predict and exact for solution: %.10f' % std2mse)

    print('mean value to relative error of mean-predict and exact for solution: %.10f' % mean2rel)
    print('std value to relative error of mean-predict and exact for solution: %.10f\n' % std2rel)

    DNN_tools.log_string('mean value to mse of mean-predict and exact for solution: %.10f' % mean2mse, log_out)
    DNN_tools.log_string('std value to mse of mean-predict and exact for solution: %.10f' % std2mse, log_out)

    DNN_tools.log_string('mean value to relative error of mean-predict and exact for solution: %.10f' % mean2rel, log_out)
    DNN_tools.log_string('std value to relative error of mean-predict and exact for solution: %.10f\n\n' % std2rel, log_out)


def print_log_mean_std2para_mse_rel(mean2mse: float = 0.01, std2mse: float = 0.01, mean2rel: float = 0.01,
                                    std2rel=0.01, log_out=None):
    print('mean value to mse of mean-predict and exact for parameter: %.10f' % mean2mse)
    print('std value to mse of mean-predict and exact for parameter: %.10f' % std2mse)

    print('mean value to relative error of mean-predict and exact for parameter: %.10f' % mean2rel)
    print('std value to relative error of mean-predict and exact for parameter: %.10f\n' % std2rel)

    DNN_tools.log_string('mean value to mse of mean-predict and exact for parameter: %.10f' % mean2mse, log_out)
    DNN_tools.log_string('std value to mse of mean-predict and exact for parameter: %.10f' % std2mse, log_out)

    DNN_tools.log_string('mean value to relative error of mean-predict and exact for parameter: %.10f' % mean2rel, log_out)
    DNN_tools.log_string('std value to relative error of mean-predict and exact for parameter: %.10f\n\n' % std2rel, log_out)


def print_log_mean_std2force_mse_rel(mean2mse: float = 0.01, std2mse: float = 0.01, mean2rel: float = 0.01,
                                     std2rel: float = 0.01, log_out=None):
    print('mean value to mse of mean-predict and exact for force-side: %.10f' % mean2mse)
    print('std value to mse of mean-predict and exact for force-side: %.10f' % std2mse)

    print('mean value to relative error of mean-predict and exact for force-side: %.10f' % mean2rel)
    print('std value to relative error of mean-predict and exact for force-side: %.10f\n' % std2rel)

    DNN_tools.log_string('mean value to mse of mean-predict and exact for force-side: %.10f' % mean2mse, log_out)
    DNN_tools.log_string('std value to mse of mean-predict and exact for force-side: %.10f' % std2mse, log_out)

    DNN_tools.log_string('mean value to relative error of mean-predict and exact for force-side: %.10f' % mean2rel, log_out)
    DNN_tools.log_string('std value to relative error of mean-predict and exact for force-side: %.10f\n\n' % std2rel, log_out)


def save_testMSE_REL2mat(Mse_data, Rel_data, name2file=None, outPath=None):
    outFile2data = '%s/testErr2%s.mat' % (outPath, name2file)
    key2mat_1 = 'mse'
    key2mat_2 = 'rel'
    scio.savemat(outFile2data, {key2mat_1: Mse_data, key2mat_2: Rel_data})


class Nets2Para_Solu(nn.Module):
    def __init__(self, dim_in=1, dim_out=1, hidden2_units=None, hidden3_units=None, hidden4_units=None,
                 name2model='DNN', Input_actName='tanh', Hidden_actName='tanh', Output_actName='linear',
                 trainable2fourier_feature=False, float_type='float32', with_gpu=False, no2gpu=0, initWB=False,
                 device='cpu'):
        super(Nets2Para_Solu, self).__init__()
        if 'NET_2HIDDEN_FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_2Hidden_FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                  actName2in=Input_actName, actName=Hidden_actName,
                                                  actName2out=Output_actName,
                                                  sigma=10.0, trainable2ff=trainable2fourier_feature,
                                                  type2float=float_type, to_gpu=with_gpu, gpu_no=no2gpu).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                  actName2in=Input_actName, actName=Hidden_actName,
                                                  actName2out=Output_actName,
                                                  sigma=2.0, trainable2ff=trainable2fourier_feature,
                                                  type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_2HIDDEN_2FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_2Hidden_2FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=10.0, trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_2FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=2.0, trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_2HIDDEN_3FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_2Hidden_3FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                                   trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_3FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=2.5, sigma3=5.0,
                                                   trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_3HIDDEN_FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_3Hidden_FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                  actName2in=Input_actName, actName=Hidden_actName,
                                                  actName2out=Output_actName,
                                                  sigma=5.0, trainable2ff=trainable2fourier_feature,
                                                  type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                  actName2in=Input_actName, actName=Hidden_actName,
                                                  actName2out=Output_actName,
                                                  sigma=2.5, trainable2ff=trainable2fourier_feature,
                                                  type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_3HIDDEN_2FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_3Hidden_2FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=10.0, trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_2FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=5, trainable2ff=trainable2fourier_feature,
                                                   type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_3HIDDEN_3FF' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_3Hidden_3FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                                   trainable2ff=trainable2fourier_feature, type2float=float_type,
                                                   to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_3FF(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                   actName2in=Input_actName, actName=Hidden_actName,
                                                   actName2out=Output_actName,
                                                   sigma1=1.0, sigma2=2.5, sigma3=5,
                                                   trainable2ff=trainable2fourier_feature, type2float=float_type,
                                                   to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_2HIDDEN_MULTISCALE_FOURIER' == str.upper(name2model):
            scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            scale2k = np.array([1, 2, 3, 4, 5])
            self.net_u = DNN2Bayes.Net_2Hidden_FourierBase(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                           actName2in=Input_actName, actName=Hidden_actName,
                                                           actName2out=Output_actName,
                                                           type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                           repeat_Highfreq=True, freq=scale2u).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_FourierBase(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                           actName2in=Input_actName, actName=Hidden_actName,
                                                           actName2out=Output_actName,
                                                           type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                           repeat_Highfreq=False, freq=scale2k).to(device)
        elif 'NET_3HIDDEN_MULTISCALE_FOURIER' == str.upper(name2model):
            scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            scale2k = np.array([1, 2, 3, 4, 5])
            self.net_u = DNN2Bayes.Net_3Hidden_FourierBase(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                           actName2in=Input_actName, actName=Hidden_actName,
                                                           actName2out=Output_actName,
                                                           type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                           repeat_Highfreq=True, freq=scale2u).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_FourierBase(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                           actName2in=Input_actName, actName=Hidden_actName,
                                                           actName2out=Output_actName,
                                                           type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                           repeat_Highfreq=False, freq=scale2k).to(device)
        elif 'NET_2HIDDEN_MULTISCALE' == str.upper(name2model):
            scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            scale2k = np.array([1, 2, 3, 4, 5])
            self.net_u = DNN2Bayes.Net_2Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=True, freq=scale2u).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=False, freq=scale2k).to(device)
        elif 'NET_3HIDDEN_MULTISCALE' == str.upper(name2model):
            scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            scale2k = np.array([1, 2, 3, 4, 5])
            self.net_u = DNN2Bayes.Net_3Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=True, freq=scale2u).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=False, freq=scale2k).to(device)
        elif 'NET_4HIDDEN_MULTISCALE' == str.upper(name2model):
            scale2u = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            scale2k = np.array([1, 2, 3, 4, 5])
            self.net_u = DNN2Bayes.Net_4Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden4_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=True, freq=scale2u).to(device)
            self.net_k = DNN2Bayes.Net_4Hidden_MultiScale(indim=dim_in, outdim=dim_out, hidden_layer=hidden4_units,
                                                          actName2in=Input_actName, actName=Hidden_actName,
                                                          actName2out=Output_actName,
                                                          type2float=float_type, to_gpu=with_gpu, gpu_no=0,
                                                          repeat_Highfreq=False, freq=scale2k).to(device)
        elif 'NET_2HIDDEN' == str.upper(name2model):
            self.net_u = BayesDNN.Net_2Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)
            self.net_k = BayesDNN.Net_2Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)
        elif 'NET_3HIDDEN' == str.upper(name2model):
            self.net_u = BayesDNN.Net_3Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)
            self.net_k = BayesDNN.Net_3Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)
        elif 'NET_2HIDDEN_FOURIER' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_2Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                            actName2in=Input_actName, actName=Hidden_actName,
                                                            actName2out=Output_actName,
                                                            type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_2Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                            actName2in=Input_actName, actName=Hidden_actName,
                                                            actName2out=Output_actName,
                                                            type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        elif 'NET_3HIDDEN_FOURIER' == str.upper(name2model):
            self.net_u = DNN2Bayes.Net_3Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden3_units,
                                                            actName2in=Input_actName, actName=Hidden_actName,
                                                            actName2out=Output_actName,
                                                            type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
            self.net_k = DNN2Bayes.Net_3Hidden_FourierBasis(indim=dim_in, outdim=dim_out, hidden_layer=hidden2_units,
                                                            actName2in=Input_actName, actName=Hidden_actName,
                                                            actName2out=Output_actName,
                                                            type2float=float_type, to_gpu=with_gpu, gpu_no=0).to(device)
        else:
            self.net_u = BayesDNN.Net_4Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden4_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)
            self.net_k = BayesDNN.Net_4Hidden(indim=dim_in, outdim=dim_out, hidden_layer=hidden4_units,
                                              actName2in=Input_actName, actName=Hidden_actName,
                                              actName2out=Output_actName,
                                              type2float=float_type, to_gpu=with_gpu, gpu_no=0, init_W_B=initWB).to(device)

    def get_networks(self):
        return self.net_u, self.net_k


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
        data["x_u"] = torch.from_numpy(xu)
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias

        xf = np.reshape(np.linspace(lb, ub, N_tr_f, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xf)
        data["x_f"] = torch.from_numpy(xf)                                       # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias

        xk = np.reshape(np.linspace(lb, ub, N_tr_k, endpoint=False, dtype=np.float32), newshape=(-1, 1))
        np.random.shuffle(xk)
        data["x_k"] = torch.from_numpy(xk)                                       # interior points
        data["y_k"] = k(data["x_k"]) + torch.randn_like(data["x_k"]) * like_std  # adding bias
    elif 'lhs' == str.lower(Rdic['opt2sampling']):
        data["x_u"] = torch.cat((torch.linspace(lb, ub, 2).view(-1, 1), (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb))
        # torch.linspace(start, end, steps) view making it a column vector, boundary points
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb                     # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias

        data["x_k"] = (ub - lb) * torch.rand(N_tr_k, 1) + lb                     # interior points
        data["y_k"] = k(data["x_k"]) + torch.randn_like(data["x_k"]) * like_std  # adding bias
    else:
        data["x_u"] = torch.cat((torch.linspace(lb, ub, 2).view(-1, 1), (ub - lb) * torch.rand(N_tr_u - 2, 1) + lb))
        # torch.linspace(start, end, steps) view making it a column vector, boundary points
        data["y_u"] = u(data["x_u"]) + torch.randn_like(data["x_u"]) * like_std  # adding bias

        data["x_f"] = (ub - lb) * torch.rand(N_tr_f, 1) + lb                     # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(data["x_f"]) * like_std  # adding bias

        data["x_k"] = (ub - lb) * torch.rand(N_tr_k, 1) + lb                     # interior points
        data["y_k"] = k(data["x_k"]) + torch.randn_like(data["x_k"]) * like_std  # adding bias

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

    mse2u_list = []
    mse2k_list = []
    mse2f_list = []
    rel2u_list = []
    rel2k_list = []
    rel2f_list = []
    num2hamilton_test = 5
    for num2ham in range(num2hamilton_test):
        model2net = Nets2Para_Solu(dim_in=Rdic['indim'], dim_out=Rdic['outdim'], hidden2_units=Rdic['Two_hidden_layer'],
                                   hidden3_units=Rdic['Three_hidden_layer'], hidden4_units=Rdic['Four_hidden_layer'],
                                   name2model=Rdic['model'], Input_actName=Rdic['act_name2Input'],
                                   Hidden_actName=Rdic['act_name2Hidden'], Output_actName='linear',
                                   trainable2fourier_feature=Rdic['trainable2ff_layer'], float_type='float32',
                                   with_gpu=False, no2gpu=0, initWB=Rdic['initWB'], device=device)
        net_u, net_k = model2net.get_networks()
        nets = [net_u, net_k]

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

        # Expected = torch.stack(log_prob_list).mean()
        # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
        # Log_Print2Bayes.print_log_validation(Expected, log_out=log_fileout)

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

        mse2u_list.append(mse2U)
        mse2k_list.append(mse2K)
        mse2f_list.append(mse2F)
        rel2u_list.append(rel2U)
        rel2k_list.append(rel2K)
        rel2f_list.append(rel2F)

        pred_list.clear()
        del pred_list

        Log_Print2Bayes.print_log_mse_rel(mse2solu=mse2U, rel2solu=rel2U, mse2para=mse2K, rel2para=rel2K, mse2force=mse2F,
                                          rel2force=rel2F, log_out=log_fileout)

        if torch.cuda.is_available():
            del model2net, params_hmc
            torch.cuda.empty_cache()

    mean_mse2u = sum(mse2u_list)/len(mse2u_list)
    std_mse2u = std2list(input_list=mse2u_list, mean2list=mean_mse2u, len2list=len(mse2u_list))
    mean_rel2u = np.sum(rel2u_list)/len(rel2u_list)
    std_rel2u = std2list(input_list=rel2u_list, mean2list=mean_rel2u, len2list=len(rel2u_list))

    mean_mse2k = sum(mse2k_list)/len(mse2k_list)
    std_mse2k = std2list(input_list=mse2k_list, mean2list=mean_mse2k, len2list=len(mse2k_list))
    mean_rel2k = sum(rel2k_list)/len(rel2k_list)
    std_rel2k = std2list(input_list=rel2k_list, mean2list=mean_rel2k, len2list=len(rel2k_list))

    mean_mse2f = sum(mse2f_list)/len(mse2f_list)
    std_mse2f = std2list(input_list=mse2f_list, mean2list=mean_mse2f, len2list=len(mse2f_list))
    mean_rel2f = sum(rel2f_list)/len(rel2f_list)
    std_rel2f = std2list(input_list=rel2f_list, mean2list=mean_rel2f, len2list=len(rel2f_list))

    print_log_mean_std2solu_mse_rel(mean2mse=mean_mse2u, std2mse=std_mse2u, mean2rel=mean_rel2u,
                                    std2rel=std_rel2u, log_out=log_fileout)

    print_log_mean_std2para_mse_rel(mean2mse=mean_mse2k, std2mse=std_mse2k, mean2rel=mean_rel2k,
                                    std2rel=std_rel2k, log_out=log_fileout)

    print_log_mean_std2force_mse_rel(mean2mse=mean_mse2f, std2mse=std_mse2f, mean2rel=mean_rel2f,
                                     std2rel=std_rel2f, log_out=log_fileout)

    save_testMSE_REL2mat(mse2u_list, rel2u_list, name2file='Solu', outPath=R['FolderName'])
    save_testMSE_REL2mat(mse2k_list, rel2k_list, name2file='Para', outPath=R['FolderName'])
    save_testMSE_REL2mat(mse2f_list, rel2f_list, name2file='Force', outPath=R['FolderName'])


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
    store_file = 'Bayes1D_Hamilton_Mean'
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
    R['PDE_type'] = 'NoninearPossion'
    # R['equa_name'] = 'PDE1'
    R['equa_name'] = 'PDE2'     # The solution have multi frequency component  sin(2*pi*x)+0.1*sin(10*pi*x)
    # R['equa_name'] = 'PDE3'     # The solution have multi frequency component  sin(5*pi*x)+0.1*sin(10*pi*x)
    # R['equa_name'] = 'PDE4'       # The solution have multi frequency component  sin(5*pi*x)+0.5*sin(10*pi*x)
    # R['equa_name'] = 'PDE5'     # The solution have multi frequency component  sin(2*pi*x)+0.2*sin(10*pi*x)

    # The setups of DNN for approximating solution, parameter and force-side
    # R['model'] = 'Net_2Hidden'
    R['model'] = 'Net_2Hidden_FF'
    # R['model'] = 'Net_2Hidden_2FF'
    # R['model'] = 'Net_2Hidden_3FF'
    # R['model'] = 'Net_2Hidden_Fourier'
    # R['model'] = 'Net_2Hidden_Multiscale'
    # R['model'] = 'Net_2Hidden_Multiscale_Fourier'

    # R['model'] = 'Net_3Hidden'
    # R['model'] = 'Net_3Hidden_FF'
    # R['model'] = 'Net_3Hidden_2FF'
    # R['model'] = 'Net_3Hidden_3FF'
    # R['model'] = 'Net_3Hidden_Fourier'
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

    R['opt2sampling'] = 'equidistance'
    # R['opt2sampling'] = 'random'

    if R['model'] == 'Net_2Hidden' or R['model'] == 'Net_3Hidden':
        R['Two_hidden_layer'] = [30, 30]
        R['Three_hidden_layer'] = [20, 20, 20]
        R['Four_hidden_layer'] = [20, 20, 20, 20]
    if R['model'] == 'Net_2Hidden_Fourier' or R['model'] == 'Net_3Hidden_Fourier':
        R['Two_hidden_layer'] = [15, 20]
        R['Three_hidden_layer'] = [15, 20, 10]
        R['Four_hidden_layer'] = [15, 20, 20, 20]
    else:
        R['Two_hidden_layer'] = [20, 30]
        R['Three_hidden_layer'] = [20, 20, 30]
        R['Four_hidden_layer'] = [10, 20, 20, 20]

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

    R['mode2update_para'] = 'PINN'
    # R['mode2update_para'] = 'Hamilton'

    R['noise_level'] = 0.2

    R['trainable2ff_layer'] = True

    R['with_gpu'] = True

    R['initWB'] = True

    if R['mode2update_para'] == 'PINN':
        R['update_lr'] = True
        R['learning_rate'] = 0.01     # this is the learning rate for optimizer in PINN  model
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
        R['update_lr'] = True
        R['learning_rate'] = 0.0025   # this is the learning rate for optimizer in Hamilton  model
        R['step2update_lr'] = 25
        R['gamma2update_lr'] = 0.85

        R['sample_num2hamilton'] = 500
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

