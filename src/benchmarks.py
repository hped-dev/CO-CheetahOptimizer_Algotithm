import numpy as np


def rastrigin(x):
    """
    Función de Rastrigin:
    f(x) = A*n + sum_{i=1}^n [ x_i^2 - A*cos(2*pi*x_i) ], A = 10
    Mínimo global: f(0, ..., 0) = 0
    Dominio típico: [-5.12, 5.12]^n
    """
    x = np.asarray(x, dtype=float)
    A = 10.0
    n = x.size
    return float(A * n + np.sum(x**2 - A * np.cos(2.0 * np.pi * x)))


def ackley(x):
    """
    Función de Ackley:
    f(x, y) = -20*exp(-0.2*sqrt(0.5*(x^2 + y^2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20
    Generalizada a n dimensiones si se pasa un vector de longitud n:
    f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + e + 20
    Mínimo global: f(0, ..., 0) = 0
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / n)
    return float(term1 + term2 + np.e + 20.0)


def sphere(x):
    """
    Función Sphere:
    f(x) = sum_{i=1}^n x_i^2
    Mínimo global: f(0, ..., 0) = 0
    Dominio teórico: (-inf, inf)^n (en experimentos se usa un rango acotado, p. ej. [-5.12, 5.12] o [-100, 100])
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rosenbrock(x):
    """
    Función de Rosenbrock:
    f(x) = sum_{i=1}^{n-1} [ 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
    Mínimo global: f(1, ..., 1) = 0
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def beale(x):
    """
    Función de Beale (definida en 2 variables: x[0], x[1]):
    f(x, y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    Mínimo global: f(3, 0.5) = 0
    Dominio típico: [-4.5, 4.5]^2
    """
    x = np.asarray(x, dtype=float)
    if x.size != 2:
        raise ValueError(f"La función de Beale requiere 2 variables, se recibieron {x.size}.")
    px, py = x[0], x[1]
    t1 = (1.5 - px + px * py) ** 2
    t2 = (2.25 - px + px * (py**2)) ** 2
    t3 = (2.625 - px + px * (py**3)) ** 2
    return float(t1 + t2 + t3)


def get_benchmark_functions(dim_nd=10, sphere_bounds=(-5.12, 5.12), rosenbrock_bounds=(-5.0, 10.0), ackley_bounds=(-5.0, 5.0)):
    """
    Devuelve un diccionario con las 5 funciones de prueba y su configuración base.
    - Beale siempre tiene dimensión 2 y dominio [-4.5, 4.5].
    - Rastrigin, Ackley, Sphere y Rosenbrock pueden probarse en dim_nd (por defecto 10, o 2 si quieres comparar en 2D).
    """
    return {
        "rastrigin": {
            "name": "Rastrigin",
            "fobj": rastrigin,
            "dim": dim_nd,
            "lb": -5.12,
            "ub": 5.12,
            "global_min": 0.0,
        },
        "ackley": {
            "name": "Ackley",
            "fobj": ackley,
            "dim": dim_nd,
            "lb": ackley_bounds[0],
            "ub": ackley_bounds[1],
            "global_min": 0.0,
        },
        "sphere": {
            "name": "Sphere",
            "fobj": sphere,
            "dim": dim_nd,
            "lb": sphere_bounds[0],
            "ub": sphere_bounds[1],
            "global_min": 0.0,
        },
        "rosenbrock": {
            "name": "Rosenbrock",
            "fobj": rosenbrock,
            "dim": dim_nd,
            "lb": rosenbrock_bounds[0],
            "ub": rosenbrock_bounds[1],
            "global_min": 0.0,
        },
        "beale": {
            "name": "Beale",
            "fobj": beale,
            "dim": 2,
            "lb": -4.5,
            "ub": 4.5,
            "global_min": 0.0,
        },
    }