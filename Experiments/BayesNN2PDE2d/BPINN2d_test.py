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
from Problems import Eqs2BayesNN2d
from utilizers import plot2Bayes
from utilizers import Log_Print2Bayes
from utilizers import DNN_tools
from utilizers import Load_data2Mat


# -div(a·grad u)=f
# where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
def model_ritz_loss2multiscale(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxy = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]

    Du_2norm= torch.reshape(torch.sum(torch.square(Du2Dxy), dim=-1), shape=[-1, 1])

    aesp = torch.cos(3.0*torch.pi*x_f[:, 0:1])*torch.sin(5.0*torch.pi*x_f[:, 1:2])

    y_f = data["y_f"]
    pred_f = torch.sum(0.5 * aesp * Du_2norm - torch.multiply(y_f, u), dim=0)
    ll = ll - 0.5 * tau_likes[1] * pred_f
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, pred_u, pred_f, aesp
        torch.cuda.empty_cache()

    return ll, output


# -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
#                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
# where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
def model_loss2multiscale(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxy = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy = Du2Dxy[:, 0:1], Du2Dxy[:, 1:2]
    DDu_Dxxy = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                   create_graph=True, retain_graph=True)[0]
    DDu_Dyxy = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                   create_graph=True, retain_graph=True)[0]

    u_xx = DDu_Dxxy[:, 0:1]
    u_yy = DDu_Dyxy[:, 1:2]

    aesp = torch.cos(3.0*torch.pi*x_f[:, 0:1])*torch.sin(5.0*torch.pi*x_f[:, 1:2])

    da_dx = -3.0*torch.pi*torch.sin(3.0*torch.pi*x_f[:, 0:1])*torch.sin(5.0*torch.pi*x_f[:, 1:2])
    da_dy = 5.0*torch.pi*torch.cos(3.0*torch.pi*x_f[:, 0:1])*torch.cos(5.0*torch.pi*x_f[:, 1:2])

    # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
    #                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
    pred_f = -1.0*(torch.multiply(da_dx, Du_Dx)+torch.multiply(aesp, u_xx)+torch.multiply(da_dy,Du_Dy)+torch.multiply(aesp,u_yy))
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, u_xx, u_yy, pred_u, pred_f, aesp, da_dx, da_dy
        torch.cuda.empty_cache()

    return ll, output


# Nonlinear Poisson Problem: lambda * Lap U + U*(U^2-1)
def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxy = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy = Du2Dxy[:, 0:1], Du2Dxy[:, 1:2]
    DDu_Dxxy = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                   create_graph=True, retain_graph=True)[0]
    DDu_Dyxy = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                   create_graph=True, retain_graph=True)[0]
    u_xx = DDu_Dxxy[:, 0:1]
    u_yy = DDu_Dyxy[:, 1:2]
    pred_f = 0.01 * (u_xx + u_yy) + u*(u**2-1)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, u_xx, u_yy, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


# Nonlinear Poisson Problem: lambda * Lap U + U*(U^2-1), the estimation for coefficient function
def model_loss_k(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_bd"]
    y_u = data["y_bd"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_in"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxy = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy = Du2Dxy[:, 0:1], Du2Dxy[:, 1:2]
    DDu_Dxxy = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                   create_graph=True, retain_graph=True)[0]
    DDu_Dyxy = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                   create_graph=True, retain_graph=True)[0]
    u_xx = DDu_Dxxy[:, 0:1]
    u_yy = DDu_Dyxy[:, 1:2]
    pred_f = 0.01 * (u_xx + u_yy) + u*(u**2-1)
    y_f = data["y_in"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, u_xx, u_yy, pred_u, pred_f
        torch.cuda.empty_cache()

    return ll, output


# Nonlinear Poisson Problem: lambda * Lap U + U*(U^2-1)
def model_loss2Poisson(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None, opt2device='cpu'):
    x_u = data["x_u"]
    y_u = data["y_u"]
    pred_u = fmodel[0](x_u, params=params_unflattened[0])
    ll = -0.5 * tau_likes[0] * ((pred_u - y_u) ** 2).sum(0)
    x_f = data["x_f"]
    # x_f = x_f.detach().requires_grad_()
    x_f = x_f.requires_grad_()
    u = fmodel[0](x_f, params=params_unflattened[0])
    Du2Dxy = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    Du_Dx, Du_Dy = Du2Dxy[:, 0:1], Du2Dxy[:, 1:2]
    DDu_Dxxy = torch.autograd.grad(Du_Dx, x_f, grad_outputs=torch.ones_like(Du_Dx),
                                   create_graph=True, retain_graph=True)[0]
    DDu_Dyxy = torch.autograd.grad(Du_Dy, x_f, grad_outputs=torch.ones_like(Du_Dy),
                                   create_graph=True, retain_graph=True)[0]
    u_xx = DDu_Dxxy[:, 0:1]
    u_yy = DDu_Dyxy[:, 1:2]
    pred_f = (u_xx + u_yy)
    y_f = data["y_f"]
    ll = ll - 0.5 * tau_likes[1] * ((pred_f - y_f) ** 2).sum(0)
    output = [pred_u, pred_f]

    if torch.cuda.is_available():
        del x_u, y_u, x_f, y_f, u, Du_Dx, Du_Dy, u_xx, u_yy, pred_u, pred_f
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

    # # hyperparameters
    # hamiltorch.set_random_seed(123)

    # using PINN mode to update the parameters of DNN or not
    if 'PINN' == Rdic['mode2update_para']:
        pinns = True
    else:
        pinns = False

    prior_std = 1
    # like_std = 0.1
    like_std = Rdic['noise_level']

    lr = Rdic['learning_rate']  # the learning rate for optimizer in PINN model or Hamilton  model
    step2update_lr = Rdic['step2update_lr']
    gamma2update_lr = Rdic['gamma2update_lr']
    open_update_lr = Rdic['update_lr']

    pde = True
    max_epoch = R['max_epoch']                 # the total iteration epoch of training for PINN
    burn = 100
    num_samples = R['sample_num2hamilton']     # the number of samplings for hamilton sampler,Hamilton 抽样次数
    L = 100

    tau_priors = 1 / prior_std**2
    tau_likes = 1 / like_std**2

    left_b = -1.0        # the left boundary of interested domain
    right_b = 1.0        # the right boundary of interested domain
    bottom_b = -1.0      # the bottom boundary of interested domain
    top_b = 1.0          # the top boundary of interested domain

    N_tr_u = 200     # the number of sampled points for dealing with solution, i.e., boundary
    N_tr_f = 2500    # the number of sampled points for dealing with force-side, i.e., governed equation for interior
    N_val = 4900     # the number of sampled points for obtaining real solution, parameter and force-side

    u_exact, f = Eqs2BayesNN2d.get_infos_2d(equa_name=Rdic['equa_name'])   # get the infos for PDE problem

    data = {}
    if 'mesh_grid' == str.lower(Rdic['opt2sampling']):
        y_point2left_right_bd = np.reshape(
            np.linspace(bottom_b, top_b, N_tr_u, endpoint=True, dtype=np.float32), newshape=(-1, 1))
        x_points2left_bd = np.ones(shape=[N_tr_u, 1]) * left_b
        x_points2right_bd = np.ones(shape=[N_tr_u, 1]) * right_b
        xy_left_b = np.concatenate([x_points2left_bd, y_point2left_right_bd], axis=-1)
        xy_right_b = np.concatenate([x_points2right_bd, y_point2left_right_bd], axis=-1)

        x_point2bottom_top_bd = np.reshape(np.linspace(left_b, right_b, N_tr_u, endpoint=True, dtype=np.float32),
                                           newshape=(-1, 1))
        y_points2bottom_bd = np.ones(shape=[N_tr_u, 1]) * bottom_b
        y_points2top_bd = np.ones(shape=[N_tr_u, 1]) * top_b

        xy_bottom_b = np.concatenate([x_point2bottom_top_bd, y_points2bottom_bd], axis=-1)
        xy_top_b = np.concatenate([x_point2bottom_top_bd, y_points2top_bd], axis=-1)

        xy_bd = np.concatenate([xy_left_b, xy_right_b, xy_bottom_b, xy_top_b], axis=0, dtype=np.float32)

        np.random.shuffle(xy_bd)
        data["x_u"] = torch.from_numpy(xy_bd)
        data["y_u"] = u_exact(data["x_u"]) + torch.randn_like(u_exact(data["x_u"])) * like_std  # adding bias

        N_tr_f_mseh = int(np.sqrt(N_tr_f)) + 1
        x_coord2in = np.reshape(np.linspace(left_b+0.001, right_b, N_tr_f_mseh, endpoint=False, dtype=np.float32),
                                newshape=(-1, 1))
        y_coord2in = np.reshape(np.linspace(bottom_b+0.001, top_b, N_tr_f_mseh, endpoint=False, dtype=np.float32),
                                newshape=(-1, 1))
        mesh_x, mesh_y = np.meshgrid(x_coord2in, y_coord2in)
        xy_in = np.concatenate([np.reshape(mesh_x, newshape=[-1, 1]), np.reshape(mesh_y, newshape=[-1, 1])], axis=-1)
        np.random.shuffle(xy_in)
        data["x_f"] = torch.from_numpy(xy_in)                                       # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(f(data["x_f"])) * like_std     # adding bias
    else:
        y_point2left_right_bd = (top_b - bottom_b) * np.random.random(size=[N_tr_u, 1]) + bottom_b
        x_points2left_bd = np.ones(shape=[N_tr_u, 1]) * left_b
        x_points2right_bd = np.ones(shape=[N_tr_u, 1]) * right_b
        xy_left_b = np.concatenate([x_points2left_bd, y_point2left_right_bd], axis=-1)
        xy_right_b = np.concatenate([x_points2right_bd, y_point2left_right_bd], axis=-1)

        x_point2bottom_top_bd = (right_b - left_b) * np.random.random(size=[N_tr_u, 1]) + left_b
        y_points2bottom_bd = np.ones(shape=[N_tr_u, 1]) * bottom_b
        y_points2top_bd = np.ones(shape=[N_tr_u, 1]) * top_b

        xy_bottom_b = np.concatenate([x_point2bottom_top_bd, y_points2bottom_bd], axis=-1)
        xy_top_b = np.concatenate([x_point2bottom_top_bd, y_points2top_bd], axis=-1)

        xy_bd = np.concatenate([xy_left_b, xy_right_b, xy_bottom_b, xy_top_b], axis=0, dtype=np.float32)
        data["x_u"] = torch.from_numpy(xy_bd)                                       # boundary points for given domain
        data["y_u"] = u_exact(data["x_u"]) + torch.randn_like(u_exact(data["x_u"])) * like_std  # adding bias

        x_rand2in = (right_b - left_b) * np.random.random(size=[N_tr_f, 1]) + left_b
        y_rand2in = (top_b - bottom_b) * np.random.random(size=[N_tr_f, 1]) + bottom_b
        xy_in = np.concatenate([x_rand2in, y_rand2in], axis=-1, dtype=np.float32)
        data["x_f"] = torch.from_numpy(xy_in)                                       # interior points
        data["y_f"] = f(data["x_f"]) + torch.randn_like(f(data["x_f"])) * like_std  # adding bias

    # exact value of solution, parameter and force-side
    data_val = {}
    if 'gene_mesh_grid' == str.lower(Rdic['opt2gene_test_data']):
        N_val2mesh = int(np.sqrt(N_val))+1
        valx_coord2in = np.reshape(np.linspace(left_b, right_b, N_val2mesh, endpoint=True, dtype=np.float32),
                                   newshape=(-1, 1))
        valy_coord2in = np.reshape(np.linspace(bottom_b, top_b, N_val2mesh, endpoint=True, dtype=np.float32),
                                   newshape=(-1, 1))
        mesh_x2val, mesh_y2val = np.meshgrid(valx_coord2in, valy_coord2in)
        val_xy_points = np.concatenate([np.reshape(mesh_x2val, newshape=[-1, 1]),
                                        np.reshape(mesh_y2val, newshape=[-1, 1])], axis=-1, dtype=np.float32)
        np.random.shuffle(val_xy_points)
        torch_val_xy = torch.from_numpy(val_xy_points)
        data_val["x_u"] = torch_val_xy

        data_val["y_u"] = u_exact(data_val["x_u"])
        data_val["x_f"] = torch_val_xy
        data_val["y_f"] = f(data_val["x_f"])
    elif 'load_matlab_data' == str.lower(Rdic['opt2gene_test_data']):
        path2test_data = '../Matlab_data2Bayes2D/'
        torch_val_xy = Load_data2Mat.get_meshData2Bayes(dim=2, data_path=path2test_data, mesh_number=7, to_torch=True,
                                                        to_float=True, to_cuda=False, gpu_no=0, use_grad2x=False)
        data_val["x_u"] = torch_val_xy

        data_val["y_u"] = u_exact(data_val["x_u"])
        data_val["x_f"] = torch_val_xy
        data_val["y_f"] = f(data_val["x_f"])
    else:
        valx_coord2in = (right_b - left_b) * np.random.random(size=[N_val, 1]) + left_b
        valy_coord2in = (top_b - bottom_b) * np.random.random(size=[N_val, 1]) + bottom_b
        val_xy_points = np.concatenate([valx_coord2in, valy_coord2in], axis=-1, dtype=np.float32)
        torch_val_xy = torch.from_numpy(val_xy_points)
        data_val["x_u"] = torch_val_xy

        data_val["y_u"] = u_exact(data_val["x_u"])
        data_val["x_f"] = torch_val_xy
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
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32',
                                          to_gpu=False, gpu_no=0).to(device)
    elif 'NET_2HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_2Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Two_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0, sigma3=10.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32', to_gpu=False,
                                          gpu_no=0).to(device)
    elif 'NET_3HIDDEN_FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                         hidden_layer=Rdic['Three_hidden_layer'],
                                         actName2in=Rdic['act_name2Input'], actName=Rdic['act_name2Hidden'],
                                         sigma=5.0, trainable2ff=Rdic['trainable2ff_layer'], type2float='float32',
                                         to_gpu=False, gpu_no=0).to(device)
    elif 'NET_3HIDDEN_2FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_2FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=5.0,
                                          trainable2ff=Rdic['trainable2ff_layer'], type2float='float32',
                                          to_gpu=False, gpu_no=0).to(device)
    elif 'NET_3HIDDEN_3FF' == str.upper(Rdic['model']):
        net_u = DNN2Bayes.Net_3Hidden_3FF(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                          hidden_layer=Rdic['Three_hidden_layer'], actName2in=Rdic['act_name2Input'],
                                          actName=Rdic['act_name2Hidden'], sigma1=1.0, sigma2=2.50, sigma3=5.0,
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
        # scale2u = np.array([1, 2, 3, 4, 5])
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
        net_u = BayesDNN.Net_3Hidden(indim=Rdic['indim'], outdim=Rdic['outdim'],
                                     hidden_layer=Rdic['Three_hidden_layer'],
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

    # sampling!! The training data is fixed for all training process? why? Can it be varied? No, it is fixed
    if 'MULTISCALE' == str.upper(Rdic['PDE_type']):
        # params_hmc = BayesNN_Utils.sample_model_bpinns(
        #     nets, data, model_loss=model_loss2multiscale, num_samples=num_samples, num_steps_per_sample=L,
        #     learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
        #     burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde, pinns=pinns,
        #     total_epochs=max_epoch)
        if 'PINN' == Rdic['mode2update_para']:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss2multiscale, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
                total_epochs=max_epoch)
        else:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss2multiscale, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, samples=params_hmc, data=data_val, model_loss=model_loss2multiscale, tau_priors=tau_priors,
            tau_likes=tau_likes, pde=pde)
    elif 'SMOOTH_K' == str.upper(Rdic['PDE_type']):
        # params_hmc = BayesNN_Utils.sample_model_bpinns(
        #     nets, data, model_loss=model_loss_k, num_samples=num_samples, num_steps_per_sample=L,
        #     learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
        #     burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde, pinns=pinns,
        #     total_epochs=max_epoch)
        if 'PINN' == Rdic['mode2update_para']:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss_k, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
                total_epochs=max_epoch)
        else:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss_k, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, samples=params_hmc, data=data_val, model_loss=model_loss_k, tau_priors=tau_priors,
            tau_likes=tau_likes, pde=pde)
    elif 'POISSON' == str.upper(Rdic['PDE_type']):
        # params_hmc = BayesNN_Utils.sample_model_bpinns(
        #     nets, data, model_loss=model_loss2Poisson, num_samples=num_samples, num_steps_per_sample=L,
        #     learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
        #     burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde, pinns=pinns,
        #     total_epochs=max_epoch)
        if 'PINN' == Rdic['mode2update_para']:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss2Poisson, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
                total_epochs=max_epoch)
        else:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss2Poisson, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, samples=params_hmc, data=data_val, model_loss=model_loss2Poisson, tau_priors=tau_priors,
            tau_likes=tau_likes, pde=pde)
    else:
        # params_hmc = BayesNN_Utils.sample_model_bpinns(
        #     nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L,
        #     learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr, gamma2change_lr=gamma2update_lr,
        #     burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde, pinns=pinns,
        #     total_epochs=max_epoch)
        if 'PINN' == Rdic['mode2update_para']:
            params_hmc, losses = BayesNN_Utils.update_paras_by_pinn(
                nets, data, model_loss=model_loss, learning_rate=lr, updatelr=open_update_lr,
                step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde,
                total_epochs=max_epoch)
        else:
            params_hmc = BayesNN_Utils.update_paras_by_hamilton(
                nets, data, model_loss=model_loss, num_samples=num_samples, num_steps_per_sample=L,
                learning_rate=lr, updatelr=open_update_lr, step2change_lr=step2update_lr,
                gamma2change_lr=gamma2update_lr,
                burn=burn, tau_priors=tau_priors, tau_likes=tau_likes, device=device, pde=pde)

        pred_list, log_prob_list = BayesNN_Utils.predict_model_bpinns(
            nets, samples=params_hmc, data=data_val, model_loss=model_loss, tau_priors=tau_priors,
            tau_likes=tau_likes, pde=pde)

    Expected = torch.stack(log_prob_list).mean()
    # print("\n Expected validation log probability: {:.3f}".format(torch.stack(log_prob_list).mean()))
    Log_Print2Bayes.print_log_validation(Expected, log_out=log_fileout)

    pred_list_u = pred_list[0].cpu().numpy()
    mean2pred_u = np.mean(pred_list_u, axis=0)

    u_val = data_val['y_u'].detach().numpy()
    x_u = data_val['x_u'].detach().numpy()

    solu2test = np.reshape(mean2pred_u, newshape=[-1, 1])
    point_abs_errs = np.abs(solu2test-u_val)
    mse2test = np.mean(np.square(solu2test - u_val), axis=0)
    rel_square = np.mean(np.square(solu2test-u_val), axis=0)/np.mean(np.square(u_val), axis=0)
    Log_Print2Bayes.print_log_errors(mse2test=mse2test, rel2test=rel_square, log_out=log_fileout)
    plot2Bayes.plot_scatter_solution2test(solu2test, test_xy=x_u, name2solu='BPINN', outPath=R['FolderName'])
    plot2Bayes.plot_scatter_solution2test(u_val, test_xy=x_u, name2solu='Exact', outPath=R['FolderName'])
    plot2Bayes.plot_scatter_solution2test(point_abs_errs, test_xy=x_u, name2solu='Absolute Error', outPath=R['FolderName'])


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
    store_file = 'Bayes2D'
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
    # R['PDE_type'] = 'NoninearPossion'
    # R['equa_name'] = 'PDE1'
    # # R['equa_name'] = 'PDE2'
    # # R['equa_name'] = 'PDE3'

    R['PDE_type'] = 'Multiscale'
    R['equa_name'] = 'Multiscale1'
    # # R['equa_name'] = 'Multiscale2'

    # R['PDE_type'] = 'Poisson'
    # R['equa_name'] = 'New_PDE1'
    # # R['equa_name'] = 'New_PDE2'

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
    FolderName = os.path.join(OUT_DIR_PDE, Module_Time)  # 路径连接
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)
    R['FolderName'] = FolderName

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  复制并保存当前文件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if platform.system() == 'Windows':
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    R['indim'] = 2
    R['outdim'] = 1

    R['opt2sampling'] = 'mesh_grid'
    # R['opt2sampling'] = 'random'

    R['opt2gene_test_data'] = 'load_matlab_data'
    # R['opt2gene_test_data'] = 'gene_mesh_grid'

    # R['Two_hidden_layer'] = [5, 10]
    # R['Two_hidden_layer'] = [8, 20]
    # R['Three_hidden_layer'] = [8, 20, 10]

    # R['Two_hidden_layer'] = [8, 25]
    # R['Three_hidden_layer'] = [8, 25, 10]

    R['Two_hidden_layer'] = [20, 30]
    R['Three_hidden_layer'] = [20, 25, 10]
    R['Four_hidden_layer'] = [20, 25, 20, 20]

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

    R['noise_level'] = 0.3
    # R['noise_level'] = 0.5

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
        R['learning_rate'] = 0.00005   # this is the learning rate for optimizer in Hamilton  model
        R['step2update_lr'] = 25
        # R['gamma2update_lr'] = 0.95
        R['gamma2update_lr'] = 0.925

        R['sample_num2hamilton'] = 600
        R['max_epoch'] = 20000

    solve_bayes(Rdic=R)
    # 隐藏层激活函数：sin 比 tanh 好
    # 采样方法：均匀采样，然后打乱，比随机采样好
    # 对于PINN模式，边界点 N_tr_u = 100, 内部点 N_tr_f = 1600 时， 效果不如 边界点 N_tr_u = 200, 内部点 N_tr_f = 2500 时
    # 对于多尺度问题, 学习率要设置的比较小，不然hamilton会崩溃。如 multiscale1，lr=0.00005 才合适；multiscale1，lr=0.000025才合适；

