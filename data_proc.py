# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 21:58
# @Author  : LTao
# @Email   : liutao310@mail.nwpu.edu.cn
# @File    : data_proc.py
# @Software: PyCharm
"""   """
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import seaborn.timeseries
import numpy as np
import csv
import math
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False


v = []
steps =[]
eps = []
v1s = []
v2s = []
v3s = []
v4s = []
v5s = []
v6s = []
a1s = []
a2s = []
a3s = []

x1s = []
x2s = []
x3s = []
x4s = []
x5s = []
x6s = []

rs = []
ts = []



MAX_EPISODES = 3000
MAX_EP_STEPS = 1000
timestep = range(MAX_EP_STEPS)

lw1 = 0.75  # linewidth 图线
lw2 = 3.0  # linewidth 数据线
fs1 = 40  # fontsize 刻度字体大小
ls1 = 35  # tick_params
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 40,
         }


I = 2  #图片保存
II = 4


# def drawing(ep, vs, r0s, a0s, xs):
def drawing(v1s, v2s, v3s, v4s, v5s, v6s):


    fig1 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax1 = fig1.add_subplot(1, 1, 1)
    plt.plot(v1s[0:999], 'r', linewidth=lw2)
    plt.plot(v2s[0:999], 'g--', linewidth=lw2)
    plt.plot(v3s[0:999], 'b-.', linewidth=lw2)
    # plt.hlines(y=1, xmin = 0, xmax =MAX_EP_STEPS*delta_t, color='k', linewidth=3,linestyles = "dotted")
    plt.hlines(y=1, xmin=0, xmax=MAX_EP_STEPS, color='k', linewidth=3, linestyles="dotted")
    # plt.xlabel('t (s)', fontsize=20)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax1.set_ylabel('线速度 /m/s', fontsize=fs1)
    plt.legend([r'$v_x$', r'$v_y$', r'$v_z$'], loc='upper right', prop=font1)
    ax1.tick_params(labelsize=ls1)
    ax1.grid(1)
    plt.savefig('images/xz/%d/-1.png'% I, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    fig2 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax2 = fig2.add_subplot(1, 1, 1)
    plt.plot(v4s[0:999], 'r', linewidth=lw2)
    plt.plot(v5s[0:999], 'g--', linewidth=lw2)
    plt.plot(v6s[0:999], 'b-.', linewidth=lw2)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    plt.xlabel('迭代步数', fontsize=fs1)
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 1))
    ax2.yaxis.offsetText.set_fontsize(ls1)
    ax2.set_ylabel('角速度 /rad/s', fontsize=fs1)
    # plt.title(u'speed', fontsize=20)
    plt.legend([r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'], loc='upper right', prop=font1)
    ax2.tick_params(labelsize=ls1)
    ax2.grid(1)
    plt.savefig('images/xz/%d/-2.png'% I, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.close()


def drawing1(a1s, a2s, a3s):


    fig3 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax3 = fig3.add_subplot(1, 1, 1)
    ax3.plot(a1s[0:999], 'sienna', linewidth=lw2)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    plt.xlabel('迭代步数', fontsize=fs1)
    ax3.set_ylabel(r'推力 $ T_p $ (N)', fontsize=fs1)
    # # plt.title(u'F', fontsize=20)
    # plt.legend(['F', 'fast', 'slow'], loc='upper right', prop=font1)
    ax3.tick_params(labelsize=ls1)
    ax3.grid(1)
    plt.savefig('images/xz/%d/-3.png'% I, dpi=300, figsize=(9, 16), bbox_inches = 'tight')
    # plt.close()
    plt.show()

    fig4 = plt.figure(figsize=(16, 10), linewidth=lw1)
    ax4 = fig4.add_subplot(1, 1, 1)
    ax4.plot(a2s[0:999], 'purple', linewidth=lw2)
    ax4.plot(a3s[0:999], 'deepskyblue',linestyle='-.', linewidth=lw2)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax4.set_ylabel(r'偏转角度 /$^\circ$', fontsize=fs1)
    # # plt.title(u'angle', fontsize=20)
    # ax4.legend([r'$\delta_r$', r'$\delta_e$'], loc='upper right', prop=font1)
    ax4.legend([r'$\alpha$', r'$\beta$'], loc='upper right', prop=font1)
    ax4.tick_params(labelsize=ls1)
    ax4.grid(1)
    plt.savefig('images/xz/%d/-4.png'% I, dpi=300, figsize=(9, 16), bbox_inches = 'tight')
    # plt.close()
    plt.show()


def drawing2(x1s, x2s, x3s, x4s, x5s, x6s):
    fig5 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax5 = fig5.add_subplot(1, 1, 1)
    plt.plot(x1s[0:999], 'r', linewidth=lw2)
    plt.plot(x2s[0:999], 'g--', linewidth=lw2)
    plt.plot(x3s[0:999], 'b-.', linewidth=lw2)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax5.set_ylabel('位置 /m', fontsize=fs1)
    plt.legend([r'$x$', r'$y$', r'$z$'], loc='upper right', prop=font1)
    ax5.tick_params(labelsize=ls1)
    ax5.grid(1)
    plt.savefig('images/xz/%d/-5.png'% I, dpi=300, figsize=(9, 16), bbox_inches='tight')
    plt.close()

    fig6 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax6 = fig6.add_subplot(1, 1, 1)
    plt.plot(x4s[0:999], 'r', linewidth=lw2)
    plt.plot(x5s[0:999], 'g--', linewidth=lw2)
    plt.plot(x6s[0:999], 'b-.', linewidth=lw2)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax6.set_ylabel('姿态 /$^\circ$', fontsize=fs1)
    plt.legend([r'$\phi$', r'$\theta$', r'$\psi$'], loc='upper right', prop=font1)
    ax6.tick_params(labelsize=ls1)
    ax6.grid(1)
    plt.savefig('images/xz/%d/-6.png'% I, dpi=300, figsize=(9, 16), bbox_inches='tight')
    plt.close()


def drawing3(rs, ts):
    fig7 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax7 = fig7.add_subplot(1, 1, 1)
    plt.plot(rs[0:999], 'orange', linewidth=lw2)
    # ax1.legend(loc='upper left', fontsize=20)
    # plt.legend(['normal', 'fast', 'slow'], loc='upper right',prop=font1)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax7.set_ylabel('奖励', fontsize=fs1)
    ax7.tick_params(labelsize=ls1)
    ax7.grid(1)
    plt.savefig('images/xz/%d/-7.png' % I, dpi=300, figsize=(9, 16), bbox_inches='tight')
    plt.close()

    fig8 = plt.figure(figsize=(16, 9), linewidth=lw1)
    ax8 = fig8.add_subplot(1, 1, 1)
    plt.plot(ts[0:999], 'c', linewidth=lw2)
    # ax1.legend(loc='upper left', fontsize=20)
    # plt.legend(['normal', 'fast', 'slow'], loc='upper right',prop=font1)
    plt.xlabel('迭代步数', fontsize=fs1)
    plt.xlim(-50, MAX_EP_STEPS + 150)
    ax8.set_ylabel('迭代耗时 /s', fontsize=fs1)
    ax8.tick_params(labelsize=ls1)
    ax8.grid(1)
    plt.savefig('images/xz/%d/-8.png' % I, dpi=300, figsize=(9, 16), bbox_inches='tight')
    plt.close()


# csv_reader = csv.reader(open('images/PPO/1-1-1/ep=843.csv'))
# csv_reader = csv.reader(open('1/ep=42.csv'))
csv_reader = csv.reader(open('2/ep=1563.csv'))
# csv_reader = csv.reader(open('3/ep=333.csv'))
# csv_reader = csv.reader(open('4/ep=424.csv'))
# csv_reader = csv.reader(open('5/ep=1966.csv'))

for row in csv_reader:
    ep = int(float(row[0]))
    eps.append(ep)
    # print(ep)
    step = int(float(row[1]))
    steps.append(step)
    v1 = float(row[2])
    v2 = float(row[3])
    v3 = float(row[4])
    v4 = float(row[5])

    # if step <35:
    #     v5 = float(row[6])*0.1
    #     v6 = float(row[7])*0.1* 0
    #
    #     a1 = float(row[8])
    #     a2 = float(row[9])*0.05
    #     a3 = float(row[10])*0.05* 0
    #
    # else:
    #     v5 = float(row[6])*0.2
    #     v6 = float(row[7])*0.2* 0
    #
    #     a1 = float(row[8])*0.5
    #     a2 = float(row[9])*0.2
    #     a3 = float(row[10])*0.2* 0
    # print(step, a1)

    v5 = float(row[6])
    v6 = float(row[7])* 0

    a1 = float(row[8])
    a2 = float(row[9])
    a3 = float(row[10])* 0

    x1 = float(row[11])
    x2 = float(row[12])* 0
    x3 = float(row[13])
    x4 = float(row[14]) * 180 / math.pi
    x5 = float(row[15]) * 180 / math.pi
    x6 = float(row[16]) * 180 / math.pi* 0

    r = float(row[17])
    t = float(row[18])




    v1s.append(v1)
    v2s.append(v2)
    v3s.append(v3)
    v4s.append(v4)
    v5s.append(v5)
    v6s.append(v6)

    a1s.append(a1)
    a2s.append(a2)
    a3s.append(a3)

    x1s.append(x1)
    x2s.append(x2)
    x3s.append(x3)
    x4s.append(x4)
    x5s.append(x5)
    x6s.append(x6)

    rs.append(r)
    ts.append(t)


# print(vs[0:999])
# plt.plot(vs[0:999])
drawing(v1s, v2s, v3s, v4s, v5s, v6s)
drawing1(a1s, a2s, a3s)
drawing2(x1s, x2s, x3s, x4s, x5s, x6s)
drawing3(rs, ts)

plt.show()
