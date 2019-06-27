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