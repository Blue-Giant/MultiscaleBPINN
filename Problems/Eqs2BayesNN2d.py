import torch
import numpy as np


def get_infos_2d(equa_name='PDE1'):
    if str.upper(equa_name) == 'PDE1':
        # 0.01*Laplace U + U(U*U-1.0) = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)
        utrue = lambda x: torch.sin(torch.pi*x[:, 0:1]) * torch.sin(torch.pi*x[:, 1:2])
        fside = lambda x: 0.01 * (-2.0*np.pi**2) * utrue(x) + utrue(x)*(torch.square(utrue(x))-1.0)
        return utrue, fside
    elif str.upper(equa_name) == 'PDE2':
        # 0.01*Laplace U + K*U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)+0.1sin(5pi*x)sin(5pi*y)
        # k(x,y,z)=xy(1-x)(1-y)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) + \
                          0.1*torch.sin(5 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])
        ktrue = lambda x: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        fside = lambda x: 0.01*(-4*(torch.pi**2)*torch.sin(2*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'PDE3':
        # 0.01*Laplace U + K*U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)+0.1sin(10pi*x)sin(10pi*y)
        # k(x,y,z)=xy(1-x)(1-y)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) + \
                          0.1*torch.sin(10 * torch.pi * x[:, 0:1])*torch.sin(10 * torch.pi * x[:, 1:2])
        ktrue = lambda x: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON1':
        # Laplace U = f
        # u(x,y)=exp(sin(pi*x))exp(sin(pi*y))
        utrue = lambda x: torch.exp(torch.sin(np.pi * x[:, 0:1])) * torch.exp(torch.sin(np.pi * x[:, 1:2]))
        fside = lambda x: (torch.pi**2)*utrue(x)*(torch.cos(np.pi*x[:, 0:1])**2-torch.sin(np.pi*x[:, 0:1]))+\
                          (torch.pi**2)*utrue(x)*(torch.cos(np.pi*x[:, 1:2])**2-torch.sin(np.pi*x[:, 1:2]))
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON2':
        # Laplace U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)+0.2sin(5pi*x)sin(5pi*y)
        utrue = lambda x: torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2]) + \
                          0.2 * torch.sin(5 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2])
        fside = lambda x: -2*(torch.pi ** 2)*torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) - \
                          10*(torch.pi ** 2)*torch.sin(5 * torch.pi * x[:, 0:1])*torch.sin(5*torch.pi*x[:, 1:2])
        return utrue, fside
    elif str.upper(equa_name) == 'LINEAR_POISSON3':
        # Laplace U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)
        utrue = lambda x: torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2])
        fside = lambda x: -2 * (torch.pi ** 2) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE1':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
        #                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
        # where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
        # cases1: a(x,y) = cos(3*pi*x)*sin(5*pi*y)   u(x,y) = sin(pi*x)*sin(pi*y)
        # da/dx = -3pi*sin(3pi*x)sin(5pi*y), du/dx = pi*cos(pi*x)sin(pi*y), ddu/dxx = -(pi**2)sin(pi*x)sin(pi*y)
        # da/dy = 5pi*cos(3pi*x)cos(5pi*y), du/dy = pi*sin(pi*x)cos(pi*y), ddu/dyy = -(pi**2)sin(pi*x)sin(pi*y)
        # fside = 3(pi**2)cos(pi*x)sin(pi*y)sin(3pi*x)sin(5pi*y) + (pi**2)sin(pi*x)sin(pi*y)cos(3pi*x)sin(5pi*y) /
        # - 5(pi**2)sin(pi*x)cos(pi*y)cos(3pi*x)cos(5pi*y) + (pi**2)sin(pi*x)sin(pi*y)cos(3pi*x)sin(5pi*y)
        utrue = lambda x: torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2])
        a = lambda x: torch.cos(3*torch.pi * x[:, 0:1]) * torch.sin(5*torch.pi * x[:, 1:2])
        da_dx = lambda x: -3.0*torch.pi*torch.sin(3.0*torch.pi*x[:, 0:1])*torch.sin(5*torch.pi * x[:, 1:2])
        du_dx = lambda x: torch.pi * torch.cos(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
        ddu_dxx = lambda x: -(torch.pi**2) * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
        da_dy = lambda x: 5*torch.pi * torch.cos(3*torch.pi * x[:, 0:1]) * torch.cos(5*torch.pi * x[:, 1:2])
        du_dy = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.cos(torch.pi * x[:, 1:2])
        ddu_dyy = lambda x: -(torch.pi**2) * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
        fside = lambda x: -(da_dx(x) * du_dx(x) + a(x) * ddu_dxx(x) + da_dy(x) * du_dy(x) + a(x) * ddu_dyy(x))

        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE2':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy] = f
        #                  -->-[(da/dx)·(du/dx) + a·ddu/dxx + (da/dy)·(du/dy)+ a·ddu/dyy] = f
        # where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
        # cases1: a(x,y) = cos(3*pi*x)*sin(5*pi*y)   u(x,y) = sin(pi*x)*sin(pi*y)+ 0.2*sin(5*pi*x)*sin(5*pi*y)
        # da/dx = -3pi*sin(3pi*x)sin(5pi*y), du/dx = pi*cos(pi*x)sin(pi*y) + 0.5pi*cos(5pi*x)sin(5pi*y),
        # ddu/dxx = -(pi**2)sin(pi*x)sin(pi*y) - 2.5(pi**2)sin(5pi*x)sin(5pi*y)
        # da/dy = 5pi*cos(3pi*x)cos(5pi*y), du/dy = pi*sin(pi*x)cos(pi*y) + 0.5pi*sin(5pi*x)cos(5pi*y),
        # ddu/dyy = -(pi**2)sin(pi*x)sin(pi*y) - 2.5(pi**2)sin(5pi*x)sin(5pi*y)
        utrue = lambda x: torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2]) + \
                          0.2*torch.sin(5*torch.pi * x[:, 0:1]) * torch.sin(5*torch.pi * x[:, 1:2])
        a = lambda x: torch.cos(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2])
        da_dx = lambda x: -3 * torch.pi * torch.sin(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2])

        du_dx = lambda x: torch.pi * torch.cos(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) + \
                          0.2* 5.0 * torch.pi * torch.cos(5 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2])

        ddu_dxx = lambda x: -(torch.pi**2) * torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2]) - \
                          0.2*25*(torch.pi**2) *torch.sin(5*torch.pi * x[:, 0:1]) * torch.sin(5*torch.pi * x[:, 1:2])

        da_dy = lambda x: 5 * torch.pi * torch.cos(3 * torch.pi * x[:, 0:1]) * torch.cos(5 * torch.pi * x[:, 1:2])

        du_dy = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.cos(torch.pi * x[:, 1:2]) + \
                          0.2* 5 * torch.pi * torch.sin(5 * torch.pi * x[:, 0:1]) * torch.cos(5 * torch.pi * x[:, 1:2])
        ddu_dyy = lambda x: -(torch.pi**2) * torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2]) - \
                          0.2*25*(torch.pi**2) * torch.sin(5*torch.pi * x[:, 0:1]) * torch.sin(5*torch.pi * x[:, 1:2])

        fside = lambda x: -(da_dx(x) * du_dx(x) + a(x) * ddu_dxx(x) + da_dy(x) * du_dy(x) + a(x) * ddu_dyy(x))
        return utrue, fside