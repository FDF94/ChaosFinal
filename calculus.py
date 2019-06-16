# Matemática
import numpy as np
from scipy import constants as sc
# Gráficos
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.style.use("seaborn-whitegrid")


def derivateA(F, Omega):
    return F*Omega


def derivateF(A, F, L, dAdt):
    result = (1-F)*dAdt + 2*L
    return result/A


def derivateOmega(A, E, F, L, Omega, dAdt, dFdt, dLdt):
    Psi = (3 * Omega**2 + 1)*(1-F) + 2*E*(1+Omega)
    term1_factor1 = 3*(L + 3*(1-F)*dAdt/2)/(A*F) + dFdt - dAdt*Psi
    term1_factor2 = L + 3*(1-F) *dAdt / 2
    term2 = 6*L*dAdt/A
    term3 = 6 * (1-F)* dAdt**2 / A
    term4 = dLdt
    term5 = 3 * (1 - F) * dFdt * Omega
    term6 = F**2 * (Psi - (1 - F)) / A**2

    return term1_factor1 * term1_factor2 + term2 + term3 - term4 - term5 - term6


def derivateL():
    return 1


def next_A_value(current_A, F, Omega, h = 0.1):
    k1 = derivateA(F,Omega)
    k2 = derivateA(F + 0.5 * h,Omega + 0.5 * h * k1)
    k3 = derivateA(F + 0.5 * h,Omega + 0.5 * h * k2)
    k4 = derivateA(F + h, Omega + h * k3)
    nextA = current_A + (k1 + 2*k2 + 2*k3 + k4) * h/6
	
    return nextA 


def next_F_value(current_F, A, L, dAdt, h = 0.1):
    k1 = derivateF(A, current_F, L, dAdt)
    k2 = derivateF(A + 0.5 * h * k1, current_F + 0.5 * h, L + 0.5 * h * k1, dAdt + 0.5 * h * k1)
    k3 = derivateF(A + 0.5 * h * k2, current_F + 0.5 * h, L + 0.5 * h * k2, dAdt + 0.5 * h * k2)
    k4 = derivateF(A + h * k3, current_F + h, L + h * k3, dAdt  + h * k3)
    nextF = current_F + (k1 + 2*k2 + 2*k3 + k4) * h/6
	
    return nextF 


def next_Omega_value(A, E, F, L, current_Omega, dAdt, dFdt, dLdt = 0, h = 0.1):
    k1 = derivateOmega(A, E, F, L, current_Omega, dAdt, dFdt, dLdt)
    k2 = derivateOmega(A + 0.5 * h * k1, E + 0.5 * h * k1, F + 0.5 * h * k1, 
        L + 0.5 * h * k1, current_Omega * 0.5 * h, dAdt + 0.5 * h * k1, 
        dFdt + 0.5 * h * k1, dLdt + 0.5 * h * k1)
    k3 = derivateOmega(A + 0.5 * h * k2, E + 0.5 * h * k2, F + 0.5 * h * k2, 
        L + 0.5 * h * k2, current_Omega * 0.5 * h, dAdt + 0.5 * h * k2, 
        dFdt + 0.5 * h * k2, dLdt + 0.5 * h * k2)
    k4 = derivateOmega(A + h * k3, E + h * k3, F + h * k3, 
        L + h * k3, current_Omega * h, dAdt + h * k3, 
        dFdt + h * k3, dLdt + h * k3)
    nextF = current_Omega + (k1 + 2*k2 + 2*k3 + k4) * h/6
	
    return nextF 


def derivateL(t):
    L0 = 0.01
    sigma = 1
    return -L0*t*np.exp(-(t**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**(5/2))


def Gaussian(t):
    L0 = 0.01
    sigma = 1
    return L0*np.exp(-(t**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma))


def main():
    t = np.arange(0, 100, 0.1)
    A = np.ones(1000, float) * 5
    E = 1
    F = np.ones(1000, float) * 0.6
    L = 0.01
    Omega = np.zeros(1000, float)
    dAdt = F*Omega
    dFdt = derivateF(A, F, L, dAdt)

    for i in range(499):
        A[i+1] = next_A_value(A[i], F[i], Omega[i])
        F[i+1] = next_F_value(F[i], A[i], L, dAdt[i])
        Omega[i+1] = next_Omega_value(A[i], E, F[i], L, Omega[i],
            dAdt[i],dFdt[i])
        dAdt[i+1] = derivateA(F[i+1], Omega[i+1])
        dFdt[i+1] = derivateF(A[i+1], F[i+1], L, dAdt[i+1])
        print(i)

    plt.plot(t, A)
    plt.plot(t, Omega)
    plt.show()
    

main()