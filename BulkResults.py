# Matemática
import numpy as np
from functions import *
# Gráficos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")
# System
import argparse

def positive_float(x):
    y=float(x)
    if y <= 0.0:
        raise argparse.ArgumentTypeError("%r not positive"%(y,))
    return y


def get_results(points_number, h, a0, f0, omega0, t, t0, L0, sigma):
    # Initialize all variables
    A = np.ones_like(t, float) * a0
    Omega = np.ones_like(t, float) * omega0
    F = np.ones_like(t, float) * f0
    L = Gaussian(t, t0, L0, sigma)

    with(np.errstate(invalid='raise', over='raise')):
        # loop over all points
        for i in range(points_number - 1):
            kA1 = derivateA(F[i], Omega[i])
            kF1 = derivateF(A[i], F[i], L[i], Omega[i])
            kO1 = derivateOmega(t[i], A[i], F[i], L[i], Omega[i], t0, L0, sigma)
            kl1 = derivateL(t[i], t0, L0, sigma)

            kA2 = derivateA(F[i] + 0.5 * h * kF1, Omega[i] + 0.5 * h * kO1)
            kF2 = derivateF(A[i] + 0.5 * h * kA1, F[i] + 0.5 * h * kF1,
                            L[i] + 0.5 * h * kl1, Omega[i] + 0.5 * h * kO1)
            kO2 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA1,
                                F[i] + 0.5 * h * kF1, L[i] + 0.5 * h * kl1,
                                Omega[i] + 0.5 * h * kO1, t0, L0, sigma)
            kl2 = derivateL(t[i] + 0.5*h, t0, L0, sigma)

            kA3 = derivateA(F[i] + 0.5 * h * kF2, Omega[i] + 0.5 * h * kO2)
            kF3 = derivateF(A[i] + 0.5 * h * kA2, F[i] + 0.5 * h * kF2,
                            L[i] + 0.5 * h * kl2, Omega[i] + 0.5 * h * kO2)
            kO3 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA2,
                                F[i] + 0.5 * h * kF2, L[i] + 0.5 * h * kl2,
                                Omega[i] + 0.5 * h * kO2, t0, L0, sigma)
            kl3 = derivateL(t[i] + 0.5*h, t0, L0, sigma)

            kA4 = derivateA(F[i] + h, Omega[i] + h * kA3)
            kF4 = derivateF(A[i] + h * kA3, F[i] + h * kF3,
                            L[i] + h * kl3, Omega[i] + h * kO3)
            kO4 = derivateOmega(t[i] + h, A[i] + h * kA3,
                                F[i] + h * kF3, L[i] + h * kl3,
                                Omega[i] + h * kO3, t0, L0, sigma)

            A[i+1] = A[i] + (kA1 + 2*kA2 + 2*kA3 + kA4) * h/6
            F[i+1] = F[i] + (kF1 + 2*kF2 + 2*kF3 + kF4) * h/6
            Omega[i+1] = Omega[i] + (kO1 + 2*kO2 + 2*kO3 + kO4) * h/6
        
        
        return A, F, L, Omega


def Bulk(t0, L0, sigma, points_number, step_size):
    A_initial_values = np.arange(3.5, 7, 0.5)
    F_initial_values = np.arange(0.1, 0.9, 0.1)
    Omega_initial_values = np.arange(-0.3, 0.3, 0.1)
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
