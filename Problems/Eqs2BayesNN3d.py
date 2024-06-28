import torch
import numpy as np


def get_infos_3d(equa_name='PDE1'):
    if str.upper(equa_name) == 'PDE1':
        # 0.01*Laplace U + U(U*U-1.0) = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)sin(pi*z)
        utrue = lambda x: torch.sin(torch.pi*x[:, 0:1]) * torch.sin(torch.pi*x[:, 1:2]) * torch.sin(torch.pi*x[:, 2:3])
        fside = lambda x: 0.01 * (-3.0*np.pi**2) * utrue(x) + utrue(x)*(torch.square(utrue(x))-1.0)
        return utrue, fside
    elif str.upper(equa_name) == 'PDE2':   # update and complete this senario
        # 0.01*Laplace U + K*U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)sin(pi*z)+0.1sin(5pi*x)sin(5pi*y)sin(5pi*z)
        # k(x,y,z)=xyz(1-x)(1-y)(1-z)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) * torch.sin(np.pi*x[:, 2:3]) + \
                          0.1*torch.sin(5 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])*\
                          torch.sin(5 * torch.pi * x[:, 2:3])
        low_freq = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2]) * torch.sin(np.pi*x[:, 2:3])
        high_freq = lambda x: torch.sin(5 * torch.pi * x[:, 0:1])*torch.sin(5 * torch.pi * x[:, 1:2])*\
                          torch.sin(5 * torch.pi * x[:, 2:3])
        ktrue = lambda x: x[:, 0:1]*(1-x[:, 0:1]) * x[:, 1:2]*(1-x[:, 1:2]) * x[:, 2:3]*(1-x[:, 2:3])
        fside = lambda x: 0.01*(-3*(torch.pi**2)*low_freq(x) - 3*2.5*(torch.pi**2)*high_freq(x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'PDE3':  # update and complete this senario
        # 0.01*Laplace U + K*U = f
        # u(x,y,z)=sin(pi*x)sin(pi*y)sin(pi*z)+0.1sin(10pi*x)sin(10pi*y)sin(10pi*z)
        # k(x,y,z)=xyz(1-x)(1-y)(1-z)
        utrue = lambda x: torch.sin(np.pi*x[:, 0:1]) * torch.sin(np.pi*x[:, 1:2])* torch.sin(np.pi*x[:, 2:3]) + \
                          0.1*torch.sin(10 * torch.pi * x[:, 0:1])*torch.sin(10 * torch.pi * x[:, 1:2])*\
                          torch.sin(10 * torch.pi * x[:, 2:3])
        low_freq = lambda x: torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2]) * torch.sin(np.pi * x[:, 2:3])
        high_freq = lambda x: torch.sin(10 * torch.pi * x[:, 0:1]) * torch.sin(10 * torch.pi * x[:, 1:2]) * \
                              torch.sin(10 * torch.pi * x[:, 2:3])
        ktrue = lambda x: x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2]) * x[:, 2:3] * (1 - x[:, 2:3])
        fside = lambda x: 0.01*(-3*(torch.pi**2)*low_freq(x) - 3*10*(torch.pi**2)*high_freq(x))+\
                          ktrue(x) * utrue(x)
        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE1':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy+ d(a·grad u)/dz] = f
        #                  -->-[(da/dx)·(du/dx)+a·ddu/dxx+(da/dy)·(du/dy)+a·ddu/dyy+(da/dz)·(du/dz)+a·ddu/dzz] = f
        # where div is divergence operator(散度算子)，grad is gradient operator(梯度算子)
        # cases1: a(x,y,z) = cos(3*pi*x)*sin(5*pi*y)*cos(3*pi*z)   u(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)
        utrue = lambda x: torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])
        a = lambda x: torch.cos(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                      torch.cos(3 * torch.pi * x[:, 2:3])
        da_dx = lambda x: -3 * torch.pi * torch.sin(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                          torch.cos(3 * torch.pi * x[:, 2:3])
        du_dx = lambda x: torch.pi * torch.cos(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])
        ddu_dxx = lambda x: -(torch.pi ** 2) * utrue(x)
        da_dy = lambda x: 5 * torch.pi * torch.cos(3 * torch.pi * x[:, 0:1]) * torch.cos(5 * torch.pi * x[:, 1:2]) * \
                          torch.cos(3 * torch.pi * x[:, 2:3])
        du_dy = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.cos(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])
        ddu_dyy = lambda x: -(torch.pi ** 2) * utrue(x)

        da_dz = lambda x: -3 * torch.pi * torch.cos(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
            torch.sin(3 * torch.pi * x[:, 2:3])
        du_dz = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.cos(torch.pi * x[:, 2:3])
        ddu_dzz = lambda x: -(torch.pi ** 2) * utrue(x)
        fside = lambda x: -(da_dx(x) * du_dx(x) + a(x) * ddu_dxx(x) + da_dy(x) * du_dy(x) + a(x) * ddu_dyy(x) + \
                            da_dz(x) * du_dz(x) + a(x) * ddu_dzz(x))
        return utrue, fside
    elif str.upper(equa_name) == 'MULTISCALE2':  # update and complete this senario
        # -div(a·grad u)=f -->-[d(a·grad u)/dx + d(a·grad u)/dy+ d(a·grad u)/dz] = f
        #                  -->-[(da/dx)·(du/dx)+a·ddu/dxx+(da/dy)·(du/dy)+a·ddu/dyy+(da/dz)·(du/dz)+a·ddu/dzz] = f
        # where div is divergence operator(散度算子)， grad is gradient operator(梯度算子)
        # cases1: a(x,y,z) = cos(3*pi*x)*sin(5*pi*y)*cos(3*pi*z)
        #         u(x,y,z) = sin(pi*x)*sin(pi*y)*sin(pi*z)+ 0.1*sin(5*pi*x)*sin(5*pi*y)*sin(5*pi*z)
        utrue = lambda x: torch.sin(torch.pi * x[:, 0:1])*torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3]) + 0.1*torch.sin(5*torch.pi * x[:, 0:1]) * \
                          torch.sin(5*torch.pi * x[:, 1:2]) * torch.sin(5*torch.pi * x[:, 2:3])

        low_freq = lambda x: torch.sin(np.pi * x[:, 0:1]) * torch.sin(np.pi * x[:, 1:2]) * torch.sin(np.pi * x[:, 2:3])
        high_freq = lambda x: torch.sin(5 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                              torch.sin(5 * torch.pi * x[:, 2:3])

        da_dx = lambda x: -3 * torch.pi * torch.sin(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                          torch.cos(3 * torch.pi * x[:, 2:3])

        du_dx = lambda x: torch.pi * torch.cos(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])+ \
                          0.5 * torch.pi * torch.cos(5 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])

        ddu_dxx = lambda x: -(torch.pi ** 2) * low_freq(x) - 2.5 * (torch.pi ** 2) * high_freq(x)

        da_dy = lambda x: 5 * torch.pi * torch.cos(3 * torch.pi * x[:, 0:1]) * torch.cos(5 * torch.pi * x[:, 1:2]) * \
                          torch.cos(3 * torch.pi * x[:, 2:3])

        du_dy = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.cos(torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3]) + \
                          0.5 * torch.pi * torch.sin(5 * torch.pi * x[:, 0:1]) * torch.cos(5 * torch.pi * x[:, 1:2]) * \
                          torch.sin(torch.pi * x[:, 2:3])

        ddu_dyy = lambda x: -(torch.pi ** 2) * low_freq(x) - 2.5 * (torch.pi ** 2) * high_freq(x)

        da_dz = lambda x: -3 * torch.pi * torch.cos(3 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                          torch.sin(3 * torch.pi * x[:, 2:3])
        du_dz = lambda x: torch.pi * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * \
                          torch.cos(torch.pi * x[:, 2:3]) + \
                          0.5 * torch.pi * torch.sin(5 * torch.pi * x[:, 0:1]) * torch.sin(5 * torch.pi * x[:, 1:2]) * \
                          torch.cos(torch.pi * x[:, 2:3])

        ddu_dzz = lambda x: -(torch.pi ** 2) * low_freq(x) - 2.5 * (torch.pi ** 2) * high_freq(x)

        fside = lambda x: -(da_dx(x) * du_dx(x) + a(x) * ddu_dxx(x) + da_dy(x) * du_dy(x) + a(x) * ddu_dyy(x) + \
                            da_dz(x) * du_dz(x) + a(x) * ddu_dzz(x))
        return utrue, fside