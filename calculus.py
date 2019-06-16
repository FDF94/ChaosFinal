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
    Psi = (3 * Omega**2 + 1)*(1-F) + 2*E(1+Omega)
    term1_factor1 = 3*(L + 3*(1-F)*dAdt/2)/(A*F) + dFdt - dAdt*Psi
    term1_factor2 = L + 3*(1-F) *dAdt / 2
    term2 = 6*L*dAdt/A
    term3 = 6 * (1-F)* dAdt**2 / A
    term4 = dLdt
    term5 = 3 * (1 - F) * dFdt * Omega
    term6 = F**2 * (Psi - (1 - F)) / A**2


def main():
    A = 1
    E = 1
    F = 1
    L = 1
    Omega = 1
    
    dAdt = derivateA(F, Omega)
    