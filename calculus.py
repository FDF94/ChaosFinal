# Matemática
import numpy as np
from scipy import constants as sc
# Gráficos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")


def derivateA(F, Omega):
    return F*Omega


def derivateF(A, F, L, Omega):
    dAdt = derivateA(F, Omega)
    result = (1-F)*dAdt + 2*L
    return result/A


def derivateOmega(t, A, E, F, L, Omega):

    dAdt = derivateA(F, Omega)
    dFdt = derivateF(A, F, L, Omega)
    dLdt = derivateL(t)
    Alpha = 1-F
    # Here if i change E by L/F final calculations change drastically
    Psi = (3 * Omega**2 + 1)* Alpha+ 2*E*(1+Omega)
    
     
    common_factor = 2/(3*F*Alpha) 
    term1_factor1 = 3*(L + 3*(Alpha)*dAdt/2)/(A*F) + dFdt - dAdt*Psi
    term1_factor2 = L + 3*Alpha *dAdt / 2
    term2 = 6*L*dAdt/A
    term3 = 6 * Alpha* dAdt**2 / A
    term4 = dLdt
    term5 = 3 * Alpha * dFdt * Omega
    term6 = F**2 * (Psi - Alpha) / A**2

    return (term1_factor1 * term1_factor2 + term2 + 
        term3 - term4 - term5 - term6)*common_factor


def derivateL(t):
    L0 = 0.01
    sigma = 1
    t0 = 10
    return -L0*(t - t0)*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**(5/2))


def Gaussian(t):
    L0 = 0.01
    sigma = 1
    t0 = 10
    return L0*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma))


def main():
    #Initialize hyperparameters
    points_number = 10000
    h = 0.01 #step size

    # Initialize all variables
    E_0 = 1
    t = np.arange(0, points_number * h, h)
    A = np.ones(points_number, float) * 6.67
    Omega = np.ones(points_number, float) * -0.17
    E = E_0*(1+Omega)
    F = np.ones(points_number, float) * 0.7
    L = Gaussian(t)

    # loop over all points
    for i in range(points_number - 1):
        kA1 = derivateA(F[i],Omega[i])
        kF1 = derivateF(A[i], F[i], L[i], Omega[i])
        kO1 = derivateOmega(t[i], A[i], E[i], F[i], L[i], Omega[i])
        kl1 = derivateL(t[i])

        kA2 = derivateA(F[i] + 0.5 * h * kF1,Omega[i] + 0.5 * h * kO1)
        kF2 = derivateF(A[i] + 0.5 * h * kA1, F[i]+ 0.5 * h * kF1, 
            L[i]+ 0.5 * h * kl1, Omega[i] + 0.5 * h * kO1)
        kO2 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA1, 
            E[i], F[i] + 0.5 * h * kF1, L[i] + 0.5 * h * kl1,
            Omega[i] + 0.5 * h * kO1)
        kl2 = derivateL(t[i] + 0.5*h)

        kA3 = derivateA(F[i] + 0.5 * h * kF2,Omega[i] + 0.5 * h * kO2)
        kF3 = derivateF(A[i] + 0.5 * h * kA2, F[i]+ 0.5 * h * kF2, 
            L[i]+ 0.5 * h * kl2, Omega[i] + 0.5 * h * kO2)
        kO3 = derivateOmega(t[i] + 0.5 * h, A[i] + 0.5 * h * kA2, 
            E[i], F[i] + 0.5 * h * kF2, L[i] + 0.5 * h * kl2,
            Omega[i] + 0.5 * h * kO2)
        kl3 = derivateL(t[i] + 0.5*h)

        kA4 = derivateA(F[i] + h, Omega[i] + h * kA3)
        kF4 = derivateF(A[i] + h * kA3, F[i]+ h * kF3, 
            L[i]+ h * kl3, Omega[i] + h * kO3)
        kO4 = derivateOmega(t[i] + h, A[i] + h * kA3, 
            E[i], F[i] + h * kF3, L[i] + h * kl3,
            Omega[i] + h * kO3)
        kl4 = derivateL(t[i] + h)
    
        A[i+1] = A[i] + (kA1 + 2*kA2 + 2*kA3 + kA4) * h/6
        F[i+1] = F[i] + (kF1 + 2*kF2 + 2*kF3 + kF4) * h/6
        Omega[i+1] = Omega[i] + (kO1 + 2*kO2 + 2*kO3 + kO4) * h/6
        E[i+1] = E_0*(1+Omega[i])
        
        print(i)
    M = (1-F)*A/2

    plt.plot(t, Omega)
    plt.plot(t, A)
    plt.plot(t, M)
    plt.show()
    

main()