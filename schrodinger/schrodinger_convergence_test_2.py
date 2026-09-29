import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


# ============================================================
# Parameters
# ============================================================

potential_a = 1.0
potential_b = 0.0

left = -5.0
right = 5.0

# Meshes used for the convergence test:
# h = 2^(-m)
m_min = 1
m_max = 8

# Very fine mesh used as the numerical reference solution
m_ref = 12


# ============================================================
# Potential
# ============================================================

def V(x):
    return potential_a * (x**2 - 1)**2 + potential_b * x


# ============================================================
# Gauss quadrature
# ============================================================

# Four Gauss points are sufficient for the polynomial
# integrands occurring here.
XI, W = leggauss(4)


# ============================================================
# Build FEM matrices
# ============================================================

def build_matrices(nodes):

    N = len(nodes) - 2

    A_diag = np.zeros(N)
    A_off = np.zeros(N - 1)

    B_diag = np.zeros(N)
    B_off = np.zeros(N - 1)

    # Loop over all elements
    for e in range(len(nodes) - 1):

        x_left = nodes[e]
        x_right = nodes[e + 1]
        h = x_right - x_left

        # Local stiffness matrix for the derivative term
        K_local = (1.0 / h) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0]
        ])

        # Local mass matrix
        M_local = (h / 6.0) * np.array([
            [2.0, 1.0],
            [1.0, 2.0]
        ])

        # Local potential matrix
        P_local = np.zeros((2, 2))

        # Map Gauss points to the current element
        gauss_points = (
            (x_left + x_right) / 2
            + (h / 2) * XI
        )

        for xi, weight in zip(gauss_points, W):

            # Local linear basis functions
            phi_left = (x_right - xi) / h
            phi_right = (xi - x_left) / h

            phi = np.array([phi_left, phi_right])

            P_local += (
                (h / 2)
                * weight
                * V(xi)
                * np.outer(phi, phi)
            )

        A_local = K_local + P_local

        # Global node numbers of this element
        global_nodes = [e, e + 1]

        # Assemble only interior degrees of freedom
        for p in range(2):

            i_global = global_nodes[p]

            if i_global == 0 or i_global == len(nodes) - 1:
                continue

            i = i_global - 1

            A_diag[i] += A_local[p, p]
            B_diag[i] += M_local[p, p]

        # Off-diagonal entry
        if 0 < e < len(nodes) - 2:

            A_off[e - 1] += A_local[0, 1]
            B_off[e - 1] += M_local[0, 1]

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


# ============================================================
# Compute the first FEM eigenpair
# ============================================================

def first_eigenpair(nodes):

    A, B = build_matrices(nodes)

    eigenvalues, eigenvectors = eigsh(
        A,
        k=1,
        M=B,
        sigma=0.0,
        which="LM"
    )

    lambda_h = eigenvalues[0]
    coefficients = eigenvectors[:, 0]

    # Add zero boundary values
    nodal_values = np.concatenate((
        [0.0],
        coefficients,
        [0.0]
    ))

    # Normalize in L2 using the mass matrix
    norm = np.sqrt(coefficients @ (B @ coefficients))

    nodal_values /= norm

    return lambda_h, nodal_values


# ============================================================
# Evaluate a piecewise linear FEM function
# ============================================================

def evaluate_fem_function(x, nodes, values):

    return np.interp(x, nodes, values)


# ============================================================
# Evaluate the derivative of a piecewise linear FEM function
# ============================================================

def evaluate_fem_derivative(x, nodes, values):

    element = np.searchsorted(nodes, x, side="right") - 1

    element = np.clip(
        element,
        0,
        len(nodes) - 2
    )

    return (
        values[element + 1] - values[element]
    ) / (
        nodes[element + 1] - nodes[element]
    )


# ============================================================
# Create a mesh with desired mesh size h
# ============================================================

def create_mesh(h):

    length = right - left

    number_of_elements = int(round(length / h))

    return np.linspace(
        left,
        right,
        number_of_elements + 1
    )


# ============================================================
# Reference solution
# ============================================================

h_ref = 2.0 ** (-m_ref)

nodes_ref = create_mesh(h_ref)

lambda_ref, u_ref = first_eigenpair(nodes_ref)

print("Reference solution")
print("------------------")
print(f"h_ref       = {h_ref:.10e}")
print(f"elements    = {len(nodes_ref) - 1}")
print(f"lambda_ref  = {lambda_ref:.12f}")
print()


# ============================================================
# Arrays for convergence results
# ============================================================

h_values = []

eigenvalue_errors = []
L2_errors = []
H1_semi_errors = []


# ============================================================
# Convergence test
# ============================================================

print("Convergence test")
print("----------------")

for m in range(m_min, m_max + 1):

    h = 2.0 ** (-m)

    nodes = create_mesh(h)

    lambda_h, u_h = first_eigenpair(nodes)

    # --------------------------------------------------------
    # Fix the sign of the eigenfunction
    # --------------------------------------------------------

    # Compare both functions on the reference mesh
    u_h_on_ref = evaluate_fem_function(
        nodes_ref,
        nodes,
        u_h
    )

    # Approximate inner product
    inner_product = np.trapz(
        u_h_on_ref * u_ref,
        nodes_ref
    )

    if inner_product < 0:
        u_h = -u_h
        u_h_on_ref = -u_h_on_ref

    # --------------------------------------------------------
    # Eigenvalue error
    # --------------------------------------------------------

    eigenvalue_error = abs(
        lambda_h - lambda_ref
    )

    # --------------------------------------------------------
    # L2 error of the eigenfunction
    # --------------------------------------------------------

    difference = u_h_on_ref - u_ref

    L2_error = np.sqrt(
        np.trapz(
            difference**2,
            nodes_ref
        )
    )

    # --------------------------------------------------------
    # L2 error of the derivative
    # --------------------------------------------------------

    # Use midpoints of the reference elements because
    # derivatives of piecewise linear functions are constant
    # on each element.
    midpoints_ref = (
        nodes_ref[:-1] + nodes_ref[1:]
    ) / 2

    u_ref_prime = evaluate_fem_derivative(
        midpoints_ref,
        nodes_ref,
        u_ref
    )

    u_h_prime = evaluate_fem_derivative(
        midpoints_ref,
        nodes,
        u_h
    )

    derivative_difference = (
        u_h_prime - u_ref_prime
    )

    element_lengths_ref = np.diff(nodes_ref)

    H1_semi_error = np.sqrt(
        np.sum(
            element_lengths_ref
            * derivative_difference**2
        )
    )

    # Store results
    h_values.append(h)

    eigenvalue_errors.append(
        eigenvalue_error
    )

    L2_errors.append(
        L2_error
    )

    H1_semi_errors.append(
        H1_semi_error
    )

    print(
        f"m = {m:2d}, "
        f"h = {h:.8f}, "
        f"N = {len(nodes) - 2:6d}, "
        f"lambda_h = {lambda_h:.10f}, "
        f"eig error = {eigenvalue_error:.4e}, "
        f"L2 error = {L2_error:.4e}, "
        f"derivative error = {H1_semi_error:.4e}"
    )


# Convert to NumPy arrays
h_values = np.array(h_values)

eigenvalue_errors = np.array(
    eigenvalue_errors
)

L2_errors = np.array(
    L2_errors
)

H1_semi_errors = np.array(
    H1_semi_errors
)


# ============================================================
# Compute convergence rates
# ============================================================

def convergence_rates(errors):

    rates = []

    for i in range(1, len(errors)):

        rate = (
            np.log(errors[i - 1] / errors[i])
            /
            np.log(h_values[i - 1] / h_values[i])
        )

        rates.append(rate)

    return np.array(rates)


eigenvalue_rates = convergence_rates(
    eigenvalue_errors
)

L2_rates = convergence_rates(
    L2_errors
)

H1_semi_rates = convergence_rates(
    H1_semi_errors
)


print("\nConvergence rates")
print("-----------------")

for i in range(1, len(h_values)):

    print(
        f"h = {h_values[i - 1]:.8f} -> "
        f"{h_values[i]:.8f}:  "
        f"eigenvalue = {eigenvalue_rates[i - 1]:.4f},  "
        f"L2 = {L2_rates[i - 1]:.4f},  "
        f"derivative = {H1_semi_rates[i - 1]:.4f}"
    )


# ============================================================
# Plot 1: Eigenvalue convergence
# ============================================================

plt.figure()

plt.loglog(
    h_values,
    eigenvalue_errors,
    marker="o",
    label=r"$|\lambda_{1,h}-\lambda_{1,\mathrm{ref}}|$"
)

C = eigenvalue_errors[0] / h_values[0]**2

plt.loglog(
    h_values,
    C * h_values**2,
    linestyle="--",
    label=r"$O(h^2)$"
)

plt.xlabel(r"Mesh size $h$")
plt.ylabel("Eigenvalue error")
plt.title("Convergence of the first eigenvalue")
plt.grid(True, which="both")
plt.legend()


# ============================================================
# Plot 2: L2 convergence of eigenfunction
# ============================================================

plt.figure()

plt.loglog(
    h_values,
    L2_errors,
    marker="o",
    label=r"$\|u_{1,h}-u_{1,\mathrm{ref}}\|_{L^2}$"
)

C = L2_errors[0] / h_values[0]**2

plt.loglog(
    h_values,
    C * h_values**2,
    linestyle="--",
    label=r"$O(h^2)$"
)

plt.xlabel(r"Mesh size $h$")
plt.ylabel(r"$L^2$ error")
plt.title("L2 convergence of the first eigenfunction")
plt.grid(True, which="both")
plt.legend()


# ============================================================
# Plot 3: L2 convergence of derivative
# ============================================================

plt.figure()

plt.loglog(
    h_values,
    H1_semi_errors,
    marker="o",
    label=r"$\|u'_{1,h}-u'_{1,\mathrm{ref}}\|_{L^2}$"
)

C = H1_semi_errors[0] / h_values[0]

plt.loglog(
    h_values,
    C * h_values,
    linestyle="--",
    label=r"$O(h)$"
)

plt.xlabel(r"Mesh size $h$")
plt.ylabel(r"$L^2$ error of derivative")
plt.title("Convergence of the derivative")
plt.grid(True, which="both")
plt.legend()


plt.show()