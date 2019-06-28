# Math
import numpy as np
from functions import get_results
from variable_types import *
# Graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")
# System
import argparse

def main(a0, f0, omega0, t0, L0, sigma, points_number, step_size):

    t = np.arange(0, points_number * step_size, step_size)
    try:
        A, F, L, Omega = get_results(points_number, step_size, a0, f0, omega0, t, t0, L0, sigma)
        M = (1-F)*A/2

        fig = plt.figure(figsize=(7, 9))
        gs = gridspec.GridSpec(nrows=4, ncols=1)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(t, Omega)
        ax0.set_title(r'$\Omega (t)$')

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(t, A)
        ax1.set_title(r'$A(t)$')

        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(t, M)
        ax2.set_title(r'$M(t)$')

        ax3 = fig.add_subplot(gs[3, 0])
        ax3.plot(t, L/F)
        ax3.set_title(r'$E(t)$')

        plt.show()
    except FloatingPointError:
        print("Number out of range, please try with a smaller step size and/or different initial conditions")


parser = argparse.ArgumentParser()
parser.add_argument("-A", "--a0", default=5,
                    help="Starting value for A(t)", type=positive_float)
parser.add_argument("-F", "--f0", default=0.5,
                    help="Starting value for F(t)", type=restricted_F)
parser.add_argument("-O", "--omega0", default=0,
                    help="Starting value for Omega(t)", type=restricted_Omega)
parser.add_argument("-s", "--step_size", default=0.01,
                    help="Step size for calculus", type=positive_float)
parser.add_argument("-pn", "--points_number", default=10000,
                    help="Number of points to calculate", type=int)
parser.add_argument("-S", "--sigma", default=10,
                    help="Width for the gaussian function", type=float)          
parser.add_argument("-L", "--L0", default=0.05,
                    help="Maximum value for the gaussian function", type=float)     
parser.add_argument("-t", "--t0", default=10,
                    help="Time when gaussian will reach its maximum value", type=float)                                                                            
args = parser.parse_args()
main(args.a0, args.f0, args.omega0, args.t0, args.L0, args.sigma, args.points_number, args.step_size)