# 目的 
Bayes PINN 进行偏微分方程的求解，包括Poisson， multi-scale Elliptic

# Remark
1. Add the network of  2Hidden_Fourier_Sub and 3Hidden_Fourier_Sub for Bayes_1d_multiscale, correct the wrong of 
   get_get_infos_1d for PDE5 in Eqs2BayesNN

2. BayesNN_Utils.sample_model_bpinns() 拆分为 BayesNN_Utils.update_paras_by_pinn()和
   BayesNN_Utils.update_paras_by_hamilton() 两个功能函数，其中BayesNN_Utils.update_paras_by_pinn()
   不仅返回网络参数，还返回训练过程中的losses。

3. hamilton 代码中，增加了学习率调整功能，stepLR模式；PINN更新参数模块增加了学习率调整功能，stepLR模式

4. BPINN2Poisson1D.py 为求解1维Poisson方程的代码，BPINN_Pre2Poisson1D.py 为装载已经训练好的模型，进行测试

5. 隐藏层激活函数：sin 比 tanh 好

6. 采样方法：均匀采样，然后打乱，比随机采样好
   
7. 对于PINN模式，边界点 N_tr_u = 100, 内部点 N_tr_f = 1600 时， 效果不如 边界点 N_tr_u = 200, 内部点 N_tr_f = 2500 时
   
8. 对于多尺度问题, 学习率要设置的比较小，不然hamilton会崩溃。如 multiscale1，lr=0.00005 才合适；multiscale1，lr=0.000025才合适