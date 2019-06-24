# Math
import numpy as np
from scipy import constants as sc
# Graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")
# System
import argparse

def restricted_F(x):
    y=float(x)
    if y <= 0.0 or y >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0)"%(y,))
    return y

def restricted_Omega(x):
    y=float(x)
    if y <= -1.0 or y >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (-1.0, 1.0)"%(y,))
    return y

def restricted_A(x):
    y=float(x)
    if y <= 0.0:
        raise argparse.ArgumentTypeError("%r not positive"%(y,))
    return y

def derivateA(F, Omega):
    return F*Omega


def derivateF(A, F, L, Omega):
    dAdt = derivateA(F, Omega)
    result = (1-F)*dAdt + 2*L
    return result/A


def derivateOmega(t, A, F, L, Omega):

    dAdt = derivateA(F, Omega)
    dFdt = derivateF(A, F, L, Omega)
    dLdt = derivateL(t)
    Alpha = 1-F
    Psi = (3 * Omega**2 + 1) * Alpha + 2*L*(1+Omega)/F

    common_factor = 2/(3*F*Alpha)
    term1_factor1 = 3*(L + 3*(Alpha)*dAdt/2)/(A*F) + dFdt - dAdt*Psi
    term1_factor2 = L + 3*Alpha * dAdt / 2
    term2 = 6*L*dAdt/A
    term3 = 6 * Alpha * dAdt**2 / A
    term4 = dLdt
    term5 = 3 * Alpha * dFdt * Omega
    term6 = F**2 * (Psi - Alpha) / A**2

    return (term1_factor1 * term1_factor2 + term2 +
            term3 - term4 - term5 - term6)*common_factor


def derivateL(t):
    L0 = 0.05
    sigma = 1
    t0 = 20
    return -L0*(t - t0)*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**(5/2))


def Gaussian(t):
    L0 = 0.05
    sigma = 1
    t0 = 20
    return L0*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma))


def get_results(points_number, h, a0, f0, omega0, t):
    # Initialize all variables
    A = np.ones_like(t, float) * a0
    Omega = np.ones_like(t, float) * omega0
    F = np.ones_like(t, float) * f0
    L = Gaussian(t)

    with(np.errstate(invalid='raise', over='raise')):
        # loop over all points
        try:
            for i in range(points_number - 1):
                kA1 = derivateA(F[i], Omega[i])
                kF1 = derivateF(A[i], F[i], L[i], Omega[i])
                kO1 = derivateOmega(t[i], A[i], F[i], L[i], Omega[i])
                kl1 = derivateL(t[i])

                kA2 = derivateA(F[i] + 0.5 * h * kF1, Omega[i] + 0.5 * h * kO1)
                kF2 = derivateF(A[i] + 0.5 * h * kA1, F[i] + 0.5 * h * kF1,
                                L[i] + 0.5 * h * kl1, Omega[i] + 0.5 * h * kO1)
                kO2 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA1,
                                    F[i] + 0.5 * h * kF1, L[i] + 0.5 * h * kl1,
                                    Omega[i] + 0.5 * h * kO1)
                kl2 = derivateL(t[i] + 0.5*h)

                kA3 = derivateA(F[i] + 0.5 * h * kF2, Omega[i] + 0.5 * h * kO2)
                kF3 = derivateF(A[i] + 0.5 * h * kA2, F[i] + 0.5 * h * kF2,
                                L[i] + 0.5 * h * kl2, Omega[i] + 0.5 * h * kO2)
                kO3 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA2,
                                    F[i] + 0.5 * h * kF2, L[i] + 0.5 * h * kl2,
                                    Omega[i] + 0.5 * h * kO2)
                kl3 = derivateL(t[i] + 0.5*h)

                kA4 = derivateA(F[i] + h, Omega[i] + h * kA3)
                kF4 = derivateF(A[i] + h * kA3, F[i] + h * kF3,
                                L[i] + h * kl3, Omega[i] + h * kO3)
                kO4 = derivateOmega(t[i] + h, A[i] + h * kA3,
                                    F[i] + h * kF3, L[i] + h * kl3,
                                    Omega[i] + h * kO3)
                kl4 = derivateL(t[i] + h)

                A[i+1] = A[i] + (kA1 + 2*kA2 + 2*kA3 + kA4) * h/6
                F[i+1] = F[i] + (kF1 + 2*kF2 + 2*kF3 + kF4) * h/6
                Omega[i+1] = Omega[i] + (kO1 + 2*kO2 + 2*kO3 + kO4) * h/6
        except FloatingPointError:
            print("omega0:" + str(omega0) + "a0:" + str(a0) + "f0:" + str(f0))
            A = np.ones_like(t, float) * a0
            Omega = np.ones_like(t, float) * omega0
            F = np.ones_like(t, float) * f0
        
        return A, F, L, Omega


def main(a0, f0, omega0, points_number, step_size):

    t = np.arange(0, points_number * step_size, step_size)

    A, F, L, Omega = get_results(points_number, step_size, a0, f0, omega0, t)

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


parser = argparse.ArgumentParser()
parser.add_argument("-A", "--a0", default=5,
                    help="Starting value for A(t)", type=restricted_A)
parser.add_argument("-F", "--f0", default=0.5,
                    help="Starting value for F(t)", type=restricted_F)
parser.add_argument("-O", "--omega0", default=0,
                    help="Starting value for Omega(t)", type=restricted_Omega)
parser.add_argument("-s", "--step_size", default=0.01,
                    help="Step size for calculus", type=restricted_A)
parser.add_argument("-pn", "--points_number", default=10000,
                    help="Number of points to calculate", type=restricted_A)                                                                            
args = parser.parse_args()
main(args.a0, args.f0, args.omega0, args.points_number, args.step_size)