import torch
import numpy as np


def get_infos_2d(equa_name='PDE1'):
    if str.upper(equa_name) == 'PDE1':
        utrue = lambda x: torch.sin(torch.pi*x[:, 0:1]) * torch.sin(torch.pi*x[:, 1:2])
        fside = lambda x: 0.01 * (-2.0*np.pi**2) * utrue(x) + utrue(x)*(torch.square(utrue(x))-1.0)
        return utrue, fside
    elif str.upper(equa_name) == 'PDE2':   # update and complete this senario
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) + \
                          0.1*torch.sin(5 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])
        fside = lambda x: 0.01*(-4*(torch.pi**2)*torch.sin(2*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)   # update this term
        return utrue, fside
    elif str.upper(equa_name) == 'PDE3':  # update and complete this senario
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) + \
                          0.1*torch.sin(10 * torch.pi * x[:, 0:1])*torch.sin(10 * torch.pi * x[:, 1:2])
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)  # update this term
        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE1':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
        #                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
        # where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
        # cases1: a(x,y) = cos(3*pi*x)*sin(5*pi*y)   u(x,y) = sin(pi*x)*sin(pi*y)
        utrue = lambda x: torch.cos(3 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-0.5*100*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE2':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
        #                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
        # where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
        # cases1: a(x,y) = cos(3*pi*x)*sin(5*pi*y)   u(x,y) = sin(pi*x)*sin(pi*y)+ 0.1*sin(5*pi*x)*sin(5*pi*y)
        utrue = lambda x: torch.cos(3 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-0.5*100*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside