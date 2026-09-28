import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags


# Potential
# We choose V = 0 because the exact eigenvalues are known.
def V(x):
    return 0


# Hat basis function beta_i
def beta(i, x, nodes):
    if nodes[i - 1] <= x <= nodes[i]:
        return (x - nodes[i - 1]) / (nodes[i] - nodes[i - 1])

    if nodes[i] <= x <= nodes[i + 1]:
        return (nodes[i + 1] - x) / (nodes[i + 1] - nodes[i])

    return 0


# Derivative of beta_i
def beta_prime(i, x, nodes):
    if nodes[i - 1] < x < nodes[i]:
        return 1 / (nodes[i] - nodes[i - 1])

    if nodes[i] < x < nodes[i + 1]:
        return -1 / (nodes[i + 1] - nodes[i])

    return 0


# Four Gauss points and weights on [-1, 1]
XI, W = leggauss(4)


def gauss_quadrature(f, left, right):
    # Map Gauss points from [-1, 1] to [left, right]
    x = (left + right) / 2 + (right - left) / 2 * XI

    return (right - left) / 2 * sum(
        weight * f(point)
        for point, weight in zip(x, W)
    )


# Compute a(beta_i, beta_j)
def a(i, j, nodes):
    integrand = lambda x: (
        beta_prime(i, x, nodes) * beta_prime(j, x, nodes)
        + V(x) * beta(i, x, nodes) * beta(j, x, nodes)
    )

    quad = 0
    for k in range(1, len(nodes)):
        quad += gauss_quadrature(integrand, nodes[k - 1], nodes[k])

    return quad


# Compute b(beta_i, beta_j)
def b(i, j, nodes):
    integrand = lambda x: beta(i, x, nodes) * beta(j, x, nodes)

    quad = 0
    for k in range(1, len(nodes)):
        quad += gauss_quadrature(integrand, nodes[k - 1], nodes[k])

    return quad


# Build sparse FEM matrices
def build_matrices(nodes):
    N = len(nodes) - 2

    A_diag = np.zeros(N)
    A_off = np.zeros(N - 1)

    B_diag = np.zeros(N)
    B_off = np.zeros(N - 1)

    for i in range(1, N + 1):
        A_diag[i - 1] = a(i, i, nodes)
        B_diag[i - 1] = b(i, i, nodes)

        if i < N:
            A_off[i - 1] = a(i, i + 1, nodes)
            B_off[i - 1] = b(i, i + 1, nodes)

    A = diags(
        [A_off, A_diag, A_off],
        offsets=[-1, 0, 1],
        format="csr"
    )

    B = diags(
        [B_off, B_diag, B_off],
        offsets=[-1, 0, 1],
        format="csr"
    )

    return A, B


# Convergence test

# Use mesh sizes
# h = 2^(-m),  m = 1, ..., m_max
m_max = 8

# Exact first eigenvalue for
# -u'' = lambda u on (-1, 1)
# u(-1) = u(1) = 0
lambda_exact = (np.pi / 2) ** 2

h_values = []
errors = []


print("Exact first eigenvalue:")
print(f"lambda_1 = {lambda_exact:.10f}")

print("\nConvergence test:\n")


for m in range(1, m_max + 1):

    # Desired mesh size
    h = 2.0 ** (-m)

    # The interval (-1, 1) has length 2.
    #
    # h = 2 / (N + 1)
    #
    # Therefore:
    #
    # N + 1 = 2 / h = 2^(m + 1)
    #
    # and hence
    #
    # N = 2^(m + 1) - 1
    #
    # where N is the number of interior nodes.
    N = 2 ** (m + 1) - 1

    # Create the uniform mesh
    nodes = np.linspace(-1, 1, N + 2)

    # Build sparse FEM matrices
    A, B = build_matrices(nodes)

    # Compute only the first eigenvalue
    eigenvalues, eigenvectors = eigsh(
        A,
        k=1,
        M=B,
        sigma=0.0,
        which="LM"
    )

    lambda_h = eigenvalues[0]

    # Absolute eigenvalue error
    error = abs(lambda_h - lambda_exact)

    h_values.append(h)
    errors.append(error)

    print(
        f"m = {m:2d}, "
        f"N = {N:5d}, "
        f"h = {h:.8f}, "
        f"lambda_h = {lambda_h:.10f}, "
        f"error = {error:.6e}"
    )


# Compute convergence rates

print("\nConvergence rates:")

for i in range(1, len(h_values)):

    rate = (
        np.log(errors[i - 1] / errors[i])
        / np.log(h_values[i - 1] / h_values[i])
    )

    print(
        f"h = {h_values[i - 1]:.8f} -> "
        f"h = {h_values[i]:.8f}: "
        f"rate = {rate:.4f}"
    )


# Plot convergence


plt.loglog(
    h_values,
    errors,
    marker="o",
    label=r"$|\lambda_{1,h} - \lambda_1|$"
)

# h^2 reference line
C = errors[0] / h_values[0] ** 2

reference_errors = [
    C * h ** 2
    for h in h_values
]

plt.loglog(
    h_values,
    reference_errors,
    linestyle="--",
    label=r"$O(h^2)$"
)

plt.xlabel(r"Mesh size $h$")
plt.ylabel(r"Error $|\lambda_{1,h} - \lambda_1|$")
plt.title("Convergence of the first FEM eigenvalue")
plt.grid(True, which="both")
plt.legend()

plt.show()
