# -*- coding: utf-8 -*-
# @Time    : 2020/12/29 23:35
# @Author  : LTao
# @Email   : liutao310@mail.nwpu.edu.cn
# @File    : AUV_env.py
# @Software: PyCharm
"""  是是是   """

import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


X_dot_u, Y_dot_v, Y_dot_r, Z_dot_w, Z_dot_q,= 0.7664, 21.6346, 0.6843, 21.6346, 2.5059
K_dot_p, M_dot_w, M_dot_q, N_dot_v, N_dot_r = 0.2278, 1.6706, 5.7029, 1.6706, 5.7029

X_u, Y_v, Z_w, K_p, M_q, N_r = 1.3884, 45.0619, 45.0619, 0.0043, 7.8206, 7.8206
X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr = 0.6942, 97.2281, 97.2281, 0.0023, 1.9446, 1.9446

x_g, y_g, z_g = 0, 0, 0.0
x_b, y_b, z_b = 0, 0, 0
g = 9.8
       #　初始化属性
m = 25     # 定义质量
Ix, Iy, Iz = 0.182, 1.82, 1.82
rG = (0, 0, 0)   # 重心位置
rB = (0, 0, 0)      # 浮心位置
# self.X_dot_u, self.Y_dot_v =

# self.Ig = np.mat([[self.Ix, 0, 0], [0, self.Iy, 0], [0, 0, self.Iz]])
W = m*g
B = W



# def AUV_ENV(F_in, tf, nsteps, delta_t, ts, v0, x0):
def AUV_ENV(F_in, delta_t, v0, x0, da):

    def param():
        # 计算M = M_RB + M_A

        def S(lamda):
            S_lamda = np.mat([[0, -lamda[2], lamda[1]], [lamda[2], 0, -lamda[1]], [-lamda[1], lamda[0], 0]])
            # print("S_lamda:", S_lamda)
            return S_lamda

        Ig = np.mat([[Ix, 0, 0], [0, Iy, 0], [0, 0, Iz]])
        M_RB_11 = np.mat(m * np.eye(3))
        # self.M_RB_12 = self.m * np.zeros([3, 3])
        M_RB_12 = -m * S(rG)
        # self.M_RB_21 = self.m * np.zeros([3, 3])
        M_RB_22 = Ig
        M_RB_21 = m * S(rG)
        M_RB_1 = np.hstack((M_RB_11, M_RB_12))
        M_RB_2 = np.hstack((M_RB_21, M_RB_22))
        M_RB = np.vstack((M_RB_1, M_RB_2))

        # M_A = -np.mat([[X_dot_u, 0, 0, 0, 0, 0], [0, Y_dot_v, 0, 0, 0, Y_dot_r], [0, 0, Z_dot_w, 0, Z_dot_q, 0], \
        #                [0, 0, 0, K_dot_p, 0, 0], [0, 0, M_dot_w, 0, M_dot_q, 0], [0, N_dot_v, 0, 0, 0, N_dot_r]])

        M_A = np.mat([[X_dot_u, 0, 0, 0, 0, 0], [0, Y_dot_v, 0, 0, 0, 0], [0, 0, Z_dot_w, 0, 0, 0], \
                       [0, 0, 0, K_dot_p, 0, 0], [0, 0, 0, 0, M_dot_q, 0], [0, 0, 0, 0, 0, N_dot_r]])

        M = M_RB + M_A
        # print("M:", M)
        # print("M.T:", np.linalg.inv(M))

        return M

    def Cor(v):
        # 计算 Coriolis and centripetal matrix C(v)
        C_RB_1 = np.zeros([3, 3])

        # C_RB_2 = np.mat([[m * (y_g * v[4] + z_g * v[5]), -m * (x_g * v[4] - v[2]), -m * (x_g * v[5] + v[1])], \
        #                  [-m * (y_g * v[3] + v[2]), m * (z_g * v[5] + x_g * v[3]), -m * (y_g * v[5] - v[0])], \
        #                  [-m * (z_g * v[3] - v[1]), -m * (z_g * v[4] + v[0]), m * (x_g * v[3] + y_g * v[4])]])
        # 此处说明：此处使用array而不是mat，原因就是数组与矩阵联合使用，最终不统一

        # C_RB_2 = np.array([[m * (y_g * v[4] + z_g * v[5]), -m * (x_g * v[4] - v[2]), -m * (x_g * v[5] + v[1])], \
        #                    [-m * (y_g * v[3] + v[2]), m * (z_g * v[5] + x_g * v[3]), -m * (y_g * v[5] - v[0])], \
        #                    [-m * (z_g * v[3] - v[1]), -m * (z_g * v[4] + v[0]), m * (x_g * v[3] + y_g * v[4])]])

        C_RB_2 = np.array([[0, -m * (x_g * v[4] - v[2]), -m * (x_g * v[5] + v[1])], \
                           [0, m * (z_g * v[5] + x_g * v[3]), -m * (y_g * v[5] - v[0])], \
                           [0, -m * (z_g * v[4] + v[0]), m * (x_g * v[3] + y_g * v[4])]])

        C_RB_3 = np.array([[-m * (y_g * v[4] + z_g * v[5]), m * (x_g * v[4] + v[2]), m * (z_g * v[3] - v[1])], \
                           [m * (x_g * v[4] - v[2]), - m * (z_g * v[5] + x_g * v[3]), m * (z_g * v[4] + v[0])], \
                           [m * (x_g * v[5] + v[1]), m * (y_g * v[5] - v[0]), -m * (x_g * v[3] + y_g * v[4])]])

        # C_RB_4 = np.array([[0, Iz * v[5], -Iy * v[4]], [-Iz * v[5], 0, Ix * v[3]], [Iy * v[4], -Ix * v[3], 0]])
        C_RB_4 = np.array([[0, Iz * v[5], -Iy * v[4]], [0, 0, Ix * v[3]], [0, -Ix * v[3], 0]])

        C_RB_12 = np.hstack([C_RB_1, C_RB_2])
        C_RB_34 = np.hstack([C_RB_3, C_RB_4])
        C_RB = np.vstack([C_RB_12, C_RB_34])

        # print("C_RB:", C_RB)
        # print(self.M)
        # C_RB = np.zeros([6, 6])

        C_A = np.zeros([6, 6])

        # S([0, 1, 2])
        C = C_RB + C_A
        # print("C:", C_RB )
        Cv = np.array(np.dot(v0, C))
        # print("Cv:", Cv)
        return Cv


    def Dam(v0):
        # 计算粘性水动力
        # Dl = np.zeros([6, 6])
        # Dn = np.zeros([6, 6])
        Dl = np.diag([X_u, Y_v, Z_w, K_p, M_q, N_r])
        Dn = np.diag(np.array([X_uu, Y_vv, Z_ww, K_pp, M_qq, N_rr]) * np.array(abs(v0)))

        # D = Dl + Dn
        D = Dl
        Dv = np.dot(v0, D)
        # print("Dv:", Dv)
        return Dv


    def G(x0):
        # 定义航行器静力
        # 定义初始值
        # phi0 = x0[3]
        phi0 = 0
        theta0 = x0[4]
        psi0 = x0[5]

        g = np.array([(W - B) * np.sin(theta0), -(W - B) * np.cos(theta0) * np.sin(psi0),
                      -(W - B) * np.cos(theta0) * np.cos(psi0), \
                      -(W * y_g - B * y_b) * np.cos(theta0) * np.cos(psi0) + (W * z_g - B * z_b) * np.cos(
                          theta0) * np.sin(psi0), \
                      (W * z_g - B * z_b) * np.sin(theta0) + (W * x_g - B * z_b) * np.cos(theta0) * np.cos(psi0), \
                      -(W * x_g - B * x_b) * np.cos(theta0) * np.sin(psi0) - (W * y_g - B * y_b) * np.sin(theta0)])
        gg = g.reshape(6, 1)
        # print("g:", gg)
        # print(type(g))
        # print("gg:", gg)
        # print(type(gg))
        return g


    def Damduct(v0, da):

        theta = np.linalg.norm(da)
        D_v = np.linalg.norm([v0[0], v0[1], v0[2]])
        # print("D_v:", D_v)
        T_LD = np.array([np.sin(da[0]), ])
        D = (-0.0005501*theta**3+ 0.02666*theta**2- 0.02568*theta+ 2.471)/ 6.25* (D_v** 2)
        L = (-4.551*10**(-6)*theta**5+ 0.004196*theta**3- 1.728*theta)/ 6.25* (D_v** 2)
        # print("D:", D)
        # Dduct = np.array([0.3949712, 0, 0, 0, 0, 0])
        # Dduct = np.array([D, 0, L, 0, 0, 0])
        Dduct = np.array([D, 0, L, 0, 0, 0])
        print("Dduct:", Dduct)

        return Dduct



    #运动学解算
    def kinematic(v0):
        phi = x0[3]
        theta = x0[4]
        psi = x0[5]

        # phi = 0
        # theta = 0
        # psi = 0
        # print("phi", phi)
        ###############

        R11 = math.cos(psi) * math.cos(theta)
        R12 = -math.sin(psi) * math.cos(phi) + math.cos(psi) * math.sin(theta) * math.sin(phi)
        R13 = math.sin(psi) * math.sin(phi) + math.cos(psi) * math.cos(phi) * math.sin(theta)

        R21 = math.sin(psi) * math.cos(theta)
        R22 = math.cos(psi) * math.cos(phi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
        R23 = -math.cos(psi) * math.sin(phi) + math.sin(theta) * math.sin(psi) * math.cos(phi)

        R31 = -math.sin(theta)
        R32 = math.cos(theta) * math.sin(phi)
        R33 = math.cos(theta) * math.cos(phi)

        # J1 = np.matrix([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])
        J1 = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
        # v1 = J1 * np.transpose(np.mat(v0[0:3]))  #需要强制转矩阵
        v1 = np.dot(J1, np.transpose(np.array(v0[0:3])))  # 需要强制转矩阵

        #################

        r11 = 1
        # r12 = math.sin(phi) * math.tan(theta)
        r12 = 0
        # r13 = math.cos(phi) * math.tan(theta)
        r13 = 0

        r21 = 0
        r22 = math.cos(phi)
        r23 = -math.sin(phi)

        r31 = 0
        r32 = math.sin(phi) / math.cos(theta)
        r33 = math.cos(phi) / math.cos(theta)

        # J2 = np.mat([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        # v2 = J2 * np.transpose(np.mat(v0[3:]))
        J2 = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        v2 = J2 * np.transpose(np.array(v0[3:]))

        J11 = J1
        J12 = np.zeros([3, 3])
        J21 = np.zeros([3, 3])
        J22 = J2

        J_1 = np.hstack([J11, J12])
        J_2 = np.hstack([J21, J22])
        J = np.vstack([J_1, J_2])

        # print('J:', J)
        # print('vv0',np.transpose(np.array(v0)))
        v_k = np.dot(J, np.transpose(np.array(v0)))
        # print("v_k", v_k)
        return v_k



    def vchicle_d(v, t):  # 定义动力学方程，求加速度
        dvdt = np.dot(F, np.linalg.inv(M))
        # print("dvdt:", dvdt)
        # print(type(dvdt))
        return dvdt



    # 定义运动学方程，求位置与姿态
    def vchicle_k(v, t):
        dxdt = kinematic(v0)
        # print("dxdt:", dxdt)
        # dxdt = v0
        return dxdt





    # F = np.array(F_in)- Cor(v0)- Dam(v0)- G(x0)- Damduct(x0)
    # F = np.array(F_in)- Dam(v0)- Damduct(v0, da)
    F = np.array(F_in) - Dam(v0)
    # print('F_in:',F_in)
    # print('Cor:', Cor(v0))
    print('Dam:', Dam(v0))
    # print('G:', G(x0))
    print('Damduct:', Damduct(v0, da))
    # #
    # print('F:', F)
    M = np.array(param())
    v = odeint(vchicle_d, v0, [0, delta_t])
    v0 = v[-1]
    # vs[i+1]= v0
    # # print(i)
    # print("v0:",v0)
    #
    x = odeint(vchicle_k, x0, [0, delta_t])
    # x[:, 3]= 0
    x0 = x[-1]
    # print('x0:', x0)
    # xs[i+1]= x0

    acc0 = np.dot(F, np.linalg.inv(M))  #加速度
    # print('a:', acc0)


    return v0, x0, acc0





if __name__ == '__main__':


    # tf = 20*np.pi  # 仿真时间*np.pi
    tf = 1000  # 仿真时间*np.pi
    nsteps = 1001  # 仿真步数
    delta_t = tf / (nsteps - 1)
    ts = np.linspace(0, tf, nsteps)

    v0 = np.zeros(6)  # 定义初始线速度与角速度
    vs = np.zeros([nsteps, 6])
    x0 = np.zeros(6)  # 定义初始位置与姿态
    xs = np.zeros([nsteps, 6])
    acc = np.zeros(6)  # 定义初始加速度
    accs = np.zeros([nsteps, 6])

    vt_info = np.zeros([nsteps, 3])

    ###############  AUV_thrust  #################################
    def auv_thrust(T_p=0, alpha=0, beta=0):

        r_p = -np.array([0.6, 0, 0])
        alpha = math.pi / 180 * alpha
        beta = math.pi / 180 * beta

        delta_x = math.cos(beta) * math.cos(alpha)
        delta_y = math.cos(beta) * math.sin(alpha)
        delta_z = -math.sin(beta)
        delta = np.array([delta_x, delta_y, delta_z])  # 直接乘不行，因为＇[ ]＇是list，　不能乘

        F = T_p * delta
        M = np.cross(r_p, F)

        tau = np.hstack([F, M])
        duct_a = np.array([alpha, beta])
        return tau, duct_a


    for i in range(nsteps-1):
        # f_test = [np.sin(i * delta_t), 0, 0, 0, 0, 0]
        # f_test = [7.91, 0, 0, 0, 0, 0]
        # f_test = [10, 0, 0, 0, 0, 0]
        # f_test = [0, 1, 0, 0, 0, 0]

        info = np.array([3, 0, 0])
        tau = auv_thrust(3, 2, 0)[0]
        da = auv_thrust(3, 2, 0)[1]
        print("da:", da)

        # info = np.array([10, 0, 15])
        # tau = auv_thrust(10, 0, 15)


        if i<25:
            info = np.array([10, 0, 0])
            tau = auv_thrust(10, 0, 0)[0]
        elif i<30:
            info = np.array([2, 10, 5])
            tau = auv_thrust(2, 10, 5)[0]
        else:
            info = np.array([2, 10, 0])
            tau = auv_thrust(2, 10, 0)[0]



        print('tau', tau)
        vt_info[i+1] = info
        f_test = tau
        # f_test = [10, 0, 0, 0, 0, 0]
        # f_test = np.sin(i * delta_t)
        # com = AUV_ENV(f_test, tf, nsteps, delta_t, ts, v0, x0)
        com = AUV_ENV(f_test, delta_t, v0, x0, da)  #(F_in, delta_t, v0, x0):
        # print("v0", com[0])
        # print('\n')

        v0 = com[0]
        print('v0:', v0)
        vs[i+1]= v0

        x0 = com[1]
        print('x0:', x0)
        xs[i+1] = x0
        # xs[i + 1] = x0 % (2 * np.pi)  # 2pi后回零

        acc = com[2]
        accs[i+1] = acc

        ss = np.hstack((com[0], com[1], com[2]))
        # print(ss[0])
        print('\n\n\n')


    plt.clf()

    font1 = {'family': 'Ubuntu Mono',
             'weight': 'normal',
             'size': 23,
             }

    fig = plt.figure(figsize = (16, 16), linewidth = 0.6)

    ax1 = fig.add_subplot(421)
    # ax1.plot(ts, vs[:, :], 'g',label = 'v0')
    ax1.plot(ts, vs[:, 0:3])
    plt.xlabel('t (s)', fontsize = 20)
    ax1.set_ylabel('v (m/s)', fontsize = 20)
    plt.title(u'L speed', fontsize = 20)
    # ax1.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax1.tick_params(labelsize=15)
    ax1.grid(1)

    ax2 = fig.add_subplot(422)
    # ax1.plot(ts, vs[:, :], 'g',label = 'v0')
    ax2.plot(ts, vs[:, 3:6]*180/math.pi)
    plt.xlabel('t (s)', fontsize = 20)
    ax2.set_ylabel('v (deg/s)', fontsize = 20)
    plt.title(u'A speed', fontsize = 20)
    # ax1.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax2.tick_params(labelsize=15)
    ax2.grid(1)


    ax3 = fig.add_subplot(423)
    # ax1.plot(ts, vs[:, :], 'g',label = 'v0')
    ax3.plot(ts, vt_info[:, 0])
    plt.xlabel('t (s)', fontsize = 20)
    ax3.set_ylabel('T_p (N)', fontsize = 20)
    plt.title(u'Thrust', fontsize = 20)
    # ax1.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax3.tick_params(labelsize=15)
    ax3.grid(1)

    ax4 = fig.add_subplot(424)
    # ax1.plot(ts, vs[:, :], 'g',label = 'v0')
    ax4.plot(ts, vt_info[:, 1:3])
    plt.xlabel('t (s)', fontsize = 20)
    ax4.set_ylabel('v (deg)', fontsize = 20)
    plt.title(u'Duct angles', fontsize = 20)
    # ax1.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax4.tick_params(labelsize=15)
    ax4.grid(1)


    ax5 = fig.add_subplot(425)
    ax5.plot(ts, xs[:, 0:3])
    plt.xlabel('t(s)', fontsize = 20)
    plt.ylabel('x (m)', fontsize = 20)
    plt.title(u'Position', fontsize = 20)
    # ax2.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax5.tick_params(labelsize=15)
    ax5.grid(1)

    ax6 = fig.add_subplot(426)
    ax6.plot(ts, (xs[:, 3:6]* 180 / math.pi)%-360)
    plt.xlabel('t(s)', fontsize = 20)
    plt.ylabel('w (deg)', fontsize = 20)
    plt.title(u'Orientation', fontsize = 20)
    # ax2.legend(loc='upper left',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    ax6.tick_params(labelsize=15)
    ax6.grid(1)

    ax7 = fig.add_subplot(427)
    # ax3.plot(ts, accs[:, 1], 'm', label = 'a0')
    ax7.plot(ts, accs[:, 0:3])
    plt.xlabel('t(s)', fontsize = 20)
    plt.ylabel('a (m/s^2)', fontsize = 20)
    plt.title(u'L accelation', fontsize = 20)
    # plt.legend(loc='upper right',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    plt.tick_params(labelsize=15)
    ax7.grid(1)

    ax8 = fig.add_subplot(428)
    # ax3.plot(ts, accs[:, 1], 'm', label = 'a0')
    ax8.plot(ts, accs[:, 3:6]* 180 / math.pi)
    plt.xlabel('t(s)', fontsize = 20)
    plt.ylabel('a (deg/s^2)', fontsize = 20)
    plt.title(u'A accelation', fontsize = 20)
    # plt.legend(loc='upper right',prop=font1)
    plt.legend(['1','2','3','4','5','6'])
    plt.tick_params(labelsize=15)
    ax8.grid(1)

    plt.subplots_adjust(left=0.1, wspace=0.25, hspace=0.5,
                        bottom=0.13, top=0.91)

    # plt.savefig('images/plot %d.png'%i, dpi=300, figsize=(10, 6))
    plt.tight_layout()
    plt.show()



    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2])
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('z')
    plt.show()




    fig = plt.figure()
    plt.plot(ts, xs[:, 0:3])
    plt.legend(['1','2','3','4','5','6'])
    plt.tick_params(labelsize=15)
    plt.grid(1)
    plt.show()


    fig = plt.figure()
    plt.plot(xs[:, 0], xs[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['1','2','3','4','5','6'])
    plt.tick_params(labelsize=15)
    plt.grid(1)
    plt.show()


    fig = plt.figure()
    plt.plot(xs[:, 0], xs[:, 2])
    plt.xlabel('x')
    plt.ylabel('z')
    plt.legend(['1','2','3','4','5','6'])
    plt.tick_params(labelsize=15)
    plt.grid(1)
    plt.show()