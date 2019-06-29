from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from functions import derivateA, derivateF, derivateOmega, derivateL, Gaussian
plt.style.use("seaborn-whitegrid")


def main():    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    A, F, Omega = np.meshgrid(np.arange(3.5, 6.5, 0.5),
                        np.arange(0.2, 0.8, 0.05),
                        np.arange(-0.05, 0.06, 0.008))
    t = 0
    t0 = 10
    L0 = 0
    sigma = 10
    L=Gaussian(t, t0, L0, sigma)

    # Make the direction data for the arrows
    a = derivateA(F, Omega)*2
    f = derivateF(A, F, np.zeros_like(Omega), Omega)*2
    o = derivateOmega(t, A, F, L, Omega, t0, L0, sigma)*2

    ax.quiver3D(A, F, Omega, a, f, o, linewidth= 0.4, 
        cmap="Blues", pivot="middle")

    ax.set_xlabel('A')
    ax.set_ylabel('F')
    ax.set_zlabel('Omega')

    plt.show()


main()
