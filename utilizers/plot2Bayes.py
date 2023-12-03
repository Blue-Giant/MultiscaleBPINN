"""
@author: LXA
 Date: 2020 年 5 月 31 日
"""
from utilizers import DNN_tools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm


def plot2u(x_val=None, u_val=None, pred_list_u=None, x_u=None, y_u=None, lb=None, ub=None, outPath=None, dataType='u'):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.plot(x_val, u_val, "r-", label="Exact")
    # plt.plot(x_val,pred_list_u.squeeze(2).T, 'b-',alpha=0.01)
    plt.plot(x_val, pred_list_u.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
    plt.fill_between(
        x_val.reshape(-1),
        pred_list_u.mean(0).squeeze().T - 2 * pred_list_u.std(0).squeeze().T,
        pred_list_u.mean(0).squeeze().T + 2 * pred_list_u.std(0).squeeze().T,
        facecolor="b",
        alpha=0.2,
        label="2 std", )

    plt.plot(x_u, y_u, "kx", markersize=5, label="Training data")
    plt.xlim([lb, ub])
    plt.legend(fontsize=10)
    plt.title('solu')

    fntmp = '%s/%s' % (outPath, dataType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot2k(x_val=None, k_val=None, pred_list_k=None, x_k=None, y_k=None, lb=None, ub=None, outPath=None, dataType='k'):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.plot(x_val, k_val, "r-", label="Exact")
    # plt.plot(x_val,pred_list_k.squeeze(2).T, 'b-',alpha=0.01)
    plt.plot(x_val, pred_list_k.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
    plt.fill_between(
        x_val.reshape(-1),
        pred_list_k.mean(0).squeeze().T - 2 * pred_list_k.std(0).squeeze().T,
        pred_list_k.mean(0).squeeze().T + 2 * pred_list_k.std(0).squeeze().T,
        facecolor="b",
        alpha=0.2,
        label="2 std",)

    plt.plot(x_k, y_k, "kx", markersize=5, label="Training data")
    plt.xlim([lb, ub])
    plt.legend(fontsize=10)
    plt.title('para')

    fntmp = '%s/%s' % (outPath, dataType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot2f(x_val=None, f_val=None, pred_list_f=None, x_f=None, y_f=None, lb=None, ub=None, outPath=None, dataType='f'):
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    plt.plot(x_val, f_val, "r-", label="Exact")
    # plt.plot(x_val,pred_list_f.squeeze(2).T, 'b-',alpha=0.01)
    plt.plot(x_val, pred_list_f.mean(0).squeeze().T, "b-", alpha=0.9, label="Mean")
    plt.fill_between(
        x_val.reshape(-1),
        pred_list_f.mean(0).squeeze().T - 2 * pred_list_f.std(0).squeeze().T,
        pred_list_f.mean(0).squeeze().T + 2 * pred_list_f.std(0).squeeze().T,
        facecolor="b",
        alpha=0.2,
        label="2 std",
    )
    plt.plot(x_f, y_f, "kx", markersize=5, label="Training data")
    plt.xlim([lb, ub])
    plt.legend(fontsize=10)
    plt.title('fside')

    fntmp = '%s/%s' % (outPath, dataType)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)


def plot_scatter_solution2test(solu2test, test_xy=None, name2solu=None, outPath=None):
    test_x_bach = np.reshape(test_xy[:, 0], newshape=[-1, 1])
    test_y_bach = np.reshape(test_xy[:, 1], newshape=[-1, 1])

    # 绘制解的3D散点图
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig)
    ax.scatter(test_x_bach, test_y_bach, solu2test, c='b', label=name2solu)

    # 绘制图例
    ax.legend(loc='best')
    # 添加坐标轴(顺序是X，Y, Z)
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_zlabel('U', fontdict={'size': 15, 'color': 'red'})

    # plt.title('solution', fontsize=15)
    fntmp = '%s/solu2%s' % (outPath, name2solu)
    DNN_tools.mySaveFig(plt, fntmp, ax=ax, isax=1, iseps=0)
