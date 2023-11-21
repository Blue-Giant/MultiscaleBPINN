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
from Problems import Eqs2BayesNN
from utilizers import plot2Bayes
from utilizers import Log_Print2Bayes
from utilizers import DNN_tools


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

    print(f"Is CUDA available?: {torch.cuda.is_available()}")
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    # hyperparameters
    hamiltorch.set_random_seed(123)

    # using PINN mode to update the parameters of DNN or not
    if 'PINN' == Rdic['mode2update_para']:
        pinns = True
    else:
        pinns = False

    prior_std = 1
    # like_std = 0.1
    like_std = Rdic['noise_level']
    if True == pinns:
        step_size = 0.01    # this is the learning rate for optimizer in PINN model
    else:
        # step_size = 0.00005  # this is the learning rate for optimizer in Hamilton  model
        # step_size = 0.0001   # this is the learning rate for optimizer in Hamilton  model
        step_size = 0.0005     # this is the learning rate for optimizer in Hamilton  model
    burn = 100
    num_samples = 500          # the number of samplings for hamilton sampler
    L = 100

    pde = True
    max_epoch = R['max_epoch']                 # the total iteration epoch of training for PINN
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

    if 'NET_2HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=5.0, trainable2ff=False,
                                         type2float='float32', to_gpu=False, gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                         actName=Rdic['act_name2Hidden'], sigma=1.0, trainable2ff=False,
                                         type2float='float32', to_gpu=False, gpu_no=0).to(device)

    elif 'NET_2HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=10.0,
                                          trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_2Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=2.5,
                                          trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0).to(device)

    elif 'NET_3HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'],
                                         actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                         sigma=5.0, trainable2ff=False, type2float='float32', to_gpu=False,
                                         gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'],
                                         actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                         sigma=2.5, trainable2ff=False, type2float='float32', to_gpu=False,
                                         gpu_no=0).to(device)

    elif 'NET_3HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=10.0,
                                          trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0).to(device)
        net_k = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5,
                                          trainable2ff=False, type2float='float32', to_gpu=False, gpu_no=0).to(device)

    elif 'NET_3HIDDEN_FOURIER' == str.upper(Rdic['model']):
        scale2u = np.array([1, 1, 2, 4, 6, 8, 10])
        scale2k = np.array([1, 2, 3, 4])
        net_u = DNN2Bayes.Net_3Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Three_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'],
                                                  actName=Rdic['act_name2Hidden'], type2float='float32', to_gpu=False,
                                                  gpu_no=0, repeat_Highfreq=True, freq=scale2u).to(device)
        net_k = DNN2Bayes.Net_3Hidden_FourierBase(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                                  hidden_layer=Rdic['Three_hidden_layer'],
                                                  actName2in=Rdic['act_name2Input'],
                                                  actName=Rdic['act_name2Hidden'], type2float='float32', to_gpu=False,
                                                  gpu_no=0, repeat_Highfreq=False, freq=scale2k).to(device)
    else:
        layer_sizes = np.concatenate([[1], Rdic['Two_hidden_layer'], [1]], axis=0)
        net_u = DNN2Bayes.Net(layer_sizes, activation=torch.tanh, sigma=5.0, trainable2sigma=False,
                              type2float='float32', to_gpu=False, gpu_no=0).to(device)
        net_k = DNN2Bayes.Net(layer_sizes, activation=torch.tanh, sigma=1.0, trainable2sigma=False,
                              type2float='float32', to_gpu=False, gpu_no=0).to(device)

    nets = [net_u, net_k]

    # sampling!! The training data is fixed for all training process? why? Can it be varied? No, it is fixed
    params_hmc = BayesNN_Utils.sample_model_bpinns(
        nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L, step_size=step_size,
        burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde, pinns=pinns,
        total_epochs=max_epoch)

    pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
        nets, params_hmc, data_val, model_loss=model_loss, tau_priors=tau_priors, tau_likes=tau_likes, pde=pde)

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    Log_Print2Bayes.print_log_validation(Expected, log_out=log_fileout)

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

    plot2Bayes.plot2u(x_val=x_val, u_val=u_val, pred_list_u=pred_list_u, x_u=x_u, y_u=y_u, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='solu2u')

    plot2Bayes.plot2k(x_val=x_val, k_val=k_val, pred_list_k=pred_list_k, x_k=x_k, y_k=y_k, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='para2k')

    plot2Bayes.plot2f(x_val=x_val, f_val=f_val, pred_list_f=pred_list_f, x_f=x_f, y_f=y_f, lb=lb, ub=ub,
                      outPath=Rdic['FolderName'], dataType='force')


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

    current_day_time = datetime.datetime.now()                 # 获取当前时间
    date_time_dir = str(current_day_time.month) + str('m_') + \
                    str(current_day_time.day) + str('d_') + str(current_day_time.hour) + str('h_') + \
                    str(current_day_time.minute) + str('m_') + str(current_day_time.second) + str('s')
    FolderName = os.path.join(OUT_DIR, date_time_dir)          # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)
    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    # The setups of Problems
    R['PDE_type'] = 'NoninearPossion'
    # R['equa_name'] = 'PDE1'
    # R['equa_name'] = 'PDE2'
    # R['equa_name'] = 'PDE3'
    R['equa_name'] = 'PDE4'

    R['indim'] = 1
    R['outdim'] = 1

    R['opt2sampling'] = 'equidistance'
    # R['opt2sampling'] = 'random'

    # The setups of DNN for approximating solution, parameter and force-side
    # R['model'] = 'Net_2Hidden_FF'
    # R['model'] = 'Net_2Hidden_2FF'

    # R['model'] = 'Net_3Hidden_FF'
    R['model'] = 'Net_3Hidden_2FF'

    # R['model'] = 'Net_3Hidden_Fourier'
    # R['model'] = 'DNN'

    # R['Two_hidden_layer'] = [5, 10]
    # R['Two_hidden_layer'] = [8, 20]
    # R['Three_hidden_layer'] = [8, 20, 10]

    # R['Two_hidden_layer'] = [8, 25]
    # R['Three_hidden_layer'] = [8, 25, 10]

    R['Two_hidden_layer'] = [10, 20]
    R['Three_hidden_layer'] = [10, 20, 10]

    # R['Two_hidden_layer'] = [15, 40]
    # R['Three_hidden_layer'] = [15, 40, 20]

    # R['act_name2Input'] = 'tanh'
    # R['act_name2Input'] = 'enh_tanh'
    # R['act_name2Input'] = 'sin'
    # R['act_name2Input'] = 'sinAddcos'
    R['act_name2Input'] = 'fourier_base'

    # R['act_name2Hidden'] = 'tanh'
    # R['act_name2Hidden'] = 'enh_tanh'
    R['act_name2Hidden'] = 'sin'
    # R['act_name2Hidden'] = 'sinAddcos'

    R['act_name2Output'] = 'linear'

    # R['mode2update_para'] = 'PINN'
    R['mode2update_para'] = 'Hamilton'

    R['noise_level'] = 0.2

    solve_bayes(Rdic=R)
    # 隐藏层激活函数：sin 比 tanh 好
    # 采样方法：均匀采样，然后打乱，比随机采样好

