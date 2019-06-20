from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


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
    L0 = 0.01
    sigma = 1
    t0 = 20
    return -L0*(t - t0)*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma**(5/2))


def Gaussian(t):
    L0 = 0.01
    sigma = 1
    t0 = 20
    return L0*np.exp(-((t - t0)**2)/(2*sigma**2))/(np.sqrt(2*np.pi*sigma))

    
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make the grid
A, F, Omega = np.meshgrid(np.arange(3.5, 6.5, 0.1),
                      np.arange(0, 1, 0.2),
                      np.arange(-0.8, 0, 0.2))
L = 0
t = 0

# Make the direction data for the arrows
a = derivateA(F, Omega)
f = derivateF(A, F, np.zeros_like(Omega), Omega)
o = derivateOmega(t, A, F, L, Omega)

# Set speed metric
speed = np.sqrt(np.abs(a) + np.abs(f) + np.abs(o))

# Set linewidth
lw = speed / speed.max()

ax.quiver(A, F, Omega, a, f, o, length=0.1, linewidth=0.2)
ax.set_xlabel('A')
ax.set_ylabel('F')
ax.set_zlabel('Omega')

plt.show()
