# Matemática
import numpy as np
from functions import get_results
from variable_types import positive_float
# Gráficos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")
# System
import argparse

def Bulk(t0, L0, sigma, points_number, step_size):
    A_initial_values = np.arange(3.5, 7, 0.5)
    F_initial_values = np.arange(0.1, 0.9, 0.1)
    Omega_initial_values = np.arange(-0.4, 0.5, 0.1)
    t = np.arange(0, points_number * step_size, step_size)

    for a0 in A_initial_values:
        for f0 in F_initial_values:
            for omega0 in Omega_initial_values:
                try:
                    A, F, L, Omega = get_results(points_number, step_size, a0, f0, omega0, t, t0, L0, sigma)

                    # M = (1-F)*A/2
                    plt.plot(t, Omega)
                except FloatingPointError:
                    print("Number out of range, please try with a smaller"+
                        "step size and/or different initial conditions")


            plt.title(r"$\Omega$; A(0)=" + str(np.around(a0, decimals=1)) 
                + r"; F(0)=" + str(np.around(f0, decimals=1)))    
            plt.savefig("$omega-a0_" + str(np.around(a0, decimals=1)) 
                + "-f0_" + str(np.around(f0, decimals=1)) + ".png")
            # plt.show()
            plt.clf()


parser = argparse.ArgumentParser()
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
Bulk(args.t0, args.L0, args.sigma, args.points_number, args.step_size)
