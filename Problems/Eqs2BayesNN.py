import torch


def get_infos_1d(equa_name='PDE1'):
    if str.upper(equa_name) == 'PDE1':
        utrue = lambda x: torch.sin(2 * torch.pi * x)
        ktrue = lambda x: 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15**2)
        fside = lambda x: 0.01 * -4 * torch.pi**2 * utrue(x) + ktrue(x) * utrue(x)
        return utrue, ktrue, fside
    elif str.upper(equa_name) == 'PDE2':
        utrue = lambda x: torch.sin(2 * torch.pi * x) + 0.1*torch.cos(10 * torch.pi * x)
        ktrue = lambda x: 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)
        fside = lambda x: 0.01*(-4*(torch.pi**2)*torch.sin(2*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, ktrue, fside
    elif str.upper(equa_name) == 'PDE3':
        utrue = lambda x: torch.sin(5 * torch.pi * x) + 0.1*torch.cos(10 * torch.pi * x)
        ktrue = lambda x: 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-10*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, ktrue, fside
    elif str.upper(equa_name) == 'PDE4':
        utrue = lambda x: torch.sin(5 * torch.pi * x) + 0.5*torch.cos(10 * torch.pi * x)
        ktrue = lambda x: 0.1 + torch.exp(-0.5 * (x - 0.5) ** 2 / 0.15 ** 2)
        fside = lambda x: 0.01*(-25*(torch.pi**2)*torch.sin(5*torch.pi*x)-0.5*100*(torch.pi**2)*torch.cos(10*torch.pi*x))+\
                          ktrue(x) * utrue(x)
        return utrue, ktrue, fside