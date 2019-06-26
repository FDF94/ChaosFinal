# Math
import numpy as np

def derivateA(F, Omega):
    return F*Omega


def derivateF(A, F, L, Omega):
    dAdt = derivateA(F, Omega)
    result = (1-F)*dAdt + 2*L
    return result/A


def derivateOmega(t, A, F, L, Omega, t0, L0, sigma):

    dAdt = derivateA(F, Omega)
    dFdt = derivateF(A, F, L, Omega)
    dLdt = derivateL(t, t0, L0, sigma)
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


def derivateL(t, t0, L0, sigma):
    return -L0*(t - t0)*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**(5/2))


def Gaussian(t, t0, L0, sigma):
    return L0*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma))