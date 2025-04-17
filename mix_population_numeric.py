#!/usr/bin/python

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import math

mpl.rcParams['lines.markerfacecolor'] = 'none'
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['font.size'] = 14


colors = ['green', 'blue', 'magenta', 'red', 'green', 'red']
labels = ['SC', 'SD', 'IC', 'ID', 'C', 'I']
markers = ['s', 'o', '>', '<', 's', 'o']

IDN = sys.argv[1]

# game constant parameters
SC = 0
SD = 1
IC = 2
ID = 3
C  = 4
I  = 5
TYPES = [SC, SD, IC, ID]

r  = 1
G  = 5
Cg = 1
Ci = 10
P0 = 10
K = 1.5


a0 = 1.
ar = 0.01
at = 0.01
Mu = 0.3

L  = 1
T  = 0.01

t_max  = 10000
step   = 0.1
t_span = (0, t_max)
t_eval = np.arange(0, t_max, step)
y0     = [0.25, 0.25, 0.25, 0.25]

# rho = np.zeros(len(TYPES), t_max)


def P(rho_C): # average payoff of each type

    payoffs = [0,0,0,0] 
    G_f = math.factorial(G-1) # term G! which is constant

    for m in range(0, G):
        coff = (G_f / (math.factorial(G-1-m) * math.factorial(m))) * math.pow(rho_C, m) * math.pow((1-rho_C), (G-1-m))
        p_sc = ((r*Cg*(m+1)) / G) - Cg
        p_sd = (r*Cg*m) / G
        p_ic = ((r*Cg*(m+1)) / G) - Cg - Ci
        p_id = ((r*Cg*m)/G) - Ci
        payoffs[SC] += coff * math.exp(p_sc/K)
        payoffs[SD] += coff * math.exp(p_sd/K)
        payoffs[IC] += coff * math.exp(p_ic/K)
        payoffs[ID] += coff * math.exp(p_id/K)

    return payoffs

# def P(x): # averate payoff
#     return [
#         Cg * ( (r/G)*( (G-1)*x+1 ) - 1 ) + P0,
#         Cg * (r/G)*(G-1)*x + P0,
#         Cg * ( (r/G)*( (G-1)*x+1 ) - 1 ) - Ci + P0,
#         Cg * (r/G)*(G-1)*x - Ci + P0
#     ]


def P_avr(rho, p):
    res = 0
    for j in TYPES:
        res += rho[j] * p[j]
    return res



def dydt(t, rho):
    
    f = np.zeros(len(TYPES))
    p = P(rho[SC]+rho[IC])
    p_avr = P_avr(rho, p)

    f[SC] = T  * rho[SC]*(p[SC]-p_avr) + ( Mu*rho[IC] - a0*at*ar*rho[IC]*rho[SC] - a0*ar*rho[ID]*rho[SC] ) * (1-T)

    f[SD] = T  * rho[SD]*(p[SD]-p_avr) + ( Mu*rho[ID] - a0*at*rho[IC]*rho[SD]    - a0*rho[ID]*rho[SD]    ) * (1-T)
    
    f[IC] = T  * rho[IC]*(p[IC]-p_avr) + (-Mu*rho[IC] + a0*ar*at*rho[IC]*rho[SC] + a0*ar*rho[ID]*rho[SC] ) * (1-T)

    f[ID] = T  * rho[ID]*(p[ID]-p_avr) + (-Mu*rho[ID] + a0*at*rho[IC]*rho[SD]    + a0*rho[ID]*rho[SD]    ) * (1-T)

    return f


def t_evol():
    sol = solve_ivp(dydt, t_span, y0, 'RK45', t_eval, atol=1e-9, rtol=1e-9)
    # sol = solve_ivp(dydt_1, t_span, y0, 'RK45', t_eval)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    for i in range(4):
        ax[0].plot(sol.t, sol.y[i], label=labels[i], color=colors[i])

    ax[1].plot(sol.t, sol.y[SC]+sol.y[IC], label='C', color='green')
    ax[1].plot(sol.t, sol.y[IC]+sol.y[ID], label='I', color='red')

    ax[0].legend(); ax[0].set_xlabel('Time')
    ax[1].legend(); ax[1].set_xlabel('Time')

    plt.show()


def r_evol(name, start, stop, dx):
    name = name + '-evol-' + IDN
    output = open(name+'.txt', 'wt')
    global r
    r = start
    param = []
    value = [[], [], [], [], [], []]
    while r<=stop :
        print(r)
        sys.stdout.flush()
        sol = solve_ivp(dydt, t_span, y0, 'RK45', t_eval, atol=1e-9, rtol=1e-9)
        param.append(r)
        value[SC].append(sol.y[SC][-1])
        value[SD].append(sol.y[SD][-1])
        value[IC].append(sol.y[IC][-1])
        value[ID].append(sol.y[ID][-1])
        value[C].append((sol.y[SC]+sol.y[IC])[-1])
        value[I].append((sol.y[IC]+sol.y[ID])[-1])
        r += dx
    for i in range(len(param)):
        for j in [SC, SD, IC, ID, C, I]:
            if value[j][i] < 0: value[j][i]=0
            if value[j][i] > 1: value[j][i]=1
        output.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(param[i], value[C][i], value[I][i], value[SC][i], value[SD][i], value[IC][i], value[ID][i]))
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(4):
        ax[0].plot(param, value[i], color=colors[i], label=labels[i], marker=markers[i])
    ax[1].plot(param, value[C], color=colors[C], label=labels[C], marker=markers[C])
    ax[1].plot(param, value[I], color=colors[I], label=labels[I], marker=markers[I])

    ax[0].set_xlabel(r'$r$')
    ax[0].set_ylabel(r'$\rho$')
    ax[0].legend()

    ax[1].set_xlabel(r'$r$')
    ax[1].set_ylabel(r'$\rho$')
    ax[1].legend()
    plt.savefig(name+'.png')



def a0_evol(name, start, stop, dx):
    name = name + '-evol-' + IDN
    output = open(name+'.txt', 'wt')
    global a0
    a0 = start
    param = []
    value = [[], [], [], [], [], []]
    while a0<=stop :
        sys.stdout.flush()
        print(a0)
        sol = solve_ivp(dydt, t_span, y0, 'RK45', t_eval)
        param.append(a0)
        value[SC].append(sol.y[SC][-1])
        value[SD].append(sol.y[SD][-1])
        value[IC].append(sol.y[IC][-1])
        value[ID].append(sol.y[ID][-1])
        value[C].append((sol.y[SC]+sol.y[IC])[-1])
        value[I].append((sol.y[IC]+sol.y[ID])[-1])
        a0 += dx
    for i in range(len(param)):
        for j in [SC, SD, IC, ID, C, I]:
            if value[j][i] < 0: value[j][i]=0
            if value[j][i] > 1: value[j][i]=1
        output.write('{0},{1},{2},{3},{4},{5},{6}\n'.format(param[i], value[C][i], value[I][i], value[SC][i], value[SD][i], value[IC][i], value[ID][i]))
        
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    for i in range(4):
        ax[0].plot(param, value[i], color=colors[i], label=labels[i], marker=markers[i])
    ax[1].plot(param, value[C], color=colors[C], label=labels[C], marker=markers[C])
    ax[1].plot(param, value[I], color=colors[I], label=labels[I], marker=markers[I])

    ax[0].set_xlabel(r'$\alpha_0$')
    ax[0].set_ylabel(r'$\rho$')
    ax[0].legend()

    ax[1].set_xlabel(r'$\alpha_0$')
    ax[1].set_ylabel(r'$\rho$')
    ax[1].legend()
    plt.savefig(name+'.png')

t_evol()
# r_evol('r', 1, 6, 0.1)
# a0_evol('a0',0,100,0.1)
