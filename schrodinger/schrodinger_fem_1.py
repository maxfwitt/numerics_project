import numpy as np
import matplotlib.pyplot as plt # plotting
from numpy.polynomial.legendre import leggauss # for integrating hat basis
from scipy.linalg import eigh # generalized eigenvalue problem


# potential
def V(x):
    return (x**2 - 1)**2


# hat basis function beta_i
def beta(i, x, nodes): # i in {1, ... N}, nodes = [x_0, x_1, ..., x_N, x_N+1]
    if nodes[i - 1] <= x <= nodes[i]:
        return (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])

    if nodes[i] <= x <= nodes[i + 1]:
        return (nodes[i + 1] - x) / (nodes[i + 1] - nodes[i])

    return 0


# derivative of beta_i
def beta_prime(i, x, nodes):
    if nodes[i - 1] < x < nodes[i]:
        return 1 / (nodes[i] - nodes[i - 1])

    if nodes[i] < x < nodes[i + 1]:
        return -1 / (nodes[i + 1] - nodes[i])

    return 0 # in the case where the derivative is not defined we also return 0 (doesn't change integral)

# four gauss points and weights on the interval [-1, 1] (default)
XI, W = leggauss(4)
# we define the outside the functions since we only need to compute them once

def gauss_quadrature(f, left, right):
    # map the gauss points from [-1, 1] to [left, right]
    x = (left + right) / 2 + (right - left) / 2 * XI

    # apply the quadrature formula
    # the factor comes from the change of variables
    return (right - left) / 2 * sum(
        weight * f(point)
        for point, weight in zip(x, W)
    )

# compute a(beta_i, beta_j)
def a(i, j, nodes):
    integrand = lambda x: (
        beta_prime(i, x, nodes) * beta_prime(j, x, nodes)
        + V(x) * beta(i, x, nodes) * beta(j, x, nodes)
    )
    # the integrand is piecewise a polynomial of deg 6
    # we know that the gauss quadrature has exactness 2n-1
    # hence we integrate with a 4 point gauss quadrature
    # on each interval and add up the integrals
    quad = 0
    for k in range(1, len(nodes)):
        quad += gauss_quadrature(integrand, nodes[k-1], nodes[k])
    return quad


# Compute b(beta_i, beta_j)
def b(i, j, nodes):
    integrand = lambda x: (
        beta(i, x, nodes) * beta(j, x, nodes)
    )
    quad = 0
    for k in range(1, len(nodes)): #k = 1, ... N + 1 since len(nodes) = N + 2
        quad += gauss_quadrature(integrand, nodes[k-1], nodes[k])
    # we could speed this up even further by using a two point gauss quadrature
    return quad


def build_matrices(nodes):
    N = len(nodes) - 2

    A = np.zeros((N, N))
    B = np.zeros((N, N))
    
    # the hat functions do not overlap when |i-j| > 1
    # in this case their product and thus a(beta_i, beta_j)
    # and b(beta_i, beta_j) vanish
    # furthermore the matrices are symmetric
    
    for i in range(1, N + 1):
        A[i - 1, i - 1] = a(i, i, nodes)
        B[i - 1, i - 1] = b(i, i, nodes)

        if i < N:
            A[i - 1, i] = A[i, i - 1] = a(i, i + 1, nodes)
            B[i - 1, i] = B[i, i - 1] = b(i, i + 1, nodes)


    return A, B


# number of interior nodes
N = 20

# partition: -1 = x0 < ... < xN+1 = 1
nodes = np.linspace(-1, 1, N + 2) # N + 2 equally spaced nodes
# we ignore the endpoints since the function we're interpolating is zero there

# build matrices
A, B = build_matrices(nodes)

# solve A c = lambda B c
eigenvalues, eigenvectors = eigh(A, B)
# eigenvectors is a matrix where the columns correspond to the eigenvectors
# the eigenvectors are the coordinates of the eigenfunctions in the spline space

# display the first five eigenvalues
print("First five eigenvalues:")
for k in range(5):
    print(f"lambda_{k + 1} = {eigenvalues[k]:.8f}")


# plot the first five eigenfunctions
for k in range(5):
    # add the zero boundary values
    # at the interior nodes we get exactly the corresponding entries of the eigenvector
    nodal_values = np.concatenate(([0], eigenvectors[:, k], [0]))

    plt.plot(
        nodes,
        nodal_values,
        marker="o",
        label=rf"$\lambda_{k + 1}={eigenvalues[k]:.4f}$"
    )

plt.xlabel("x")
plt.ylabel(r"$u_h(x)$")
plt.title("First five finite-element eigenfunctions")
plt.grid()
plt.legend()
plt.show()