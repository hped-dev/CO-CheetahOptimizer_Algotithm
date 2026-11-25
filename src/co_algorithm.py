import numpy as np


def cheetah_optimizer(
    fobj,
    lb,
    ub,
    dim,
    n=6,
    m=2,
    max_fes=None,
    alpha_scale=0.0001,
    r_check_scale=1.0,
    use_improvement_rule=False,
    seed=None,
):
    """
    Adaptación a Python de Cheetah Optimizer (CO_VectorBased.m).
    Minimiza fobj(x) en [lb, ub]^dim.
    """
    rng = np.random.default_rng(seed)

    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    if lb_arr.ndim == 0:
        lb_arr = np.full(dim, float(lb_arr))
    if ub_arr.ndim == 0:
        ub_arr = np.full(dim, float(ub_arr))

    if max_fes is None:
        max_fes = dim * 10000

    # Población inicial
    pop_pos = lb_arr + rng.random((n, dim)) * (ub_arr - lb_arr)
    pop_cost = np.array([fobj(ind) for ind in pop_pos], dtype=float)
    fes = n

    best_idx = int(np.argmin(pop_cost))
    best_sol_pos = pop_pos[best_idx].copy()
    best_sol_cost = float(pop_cost[best_idx])

    # Posiciones iniciales (hogar) y mejor presa global
    home_pos = pop_pos.copy()
    x_best_pos = best_sol_pos.copy()
    x_best_cost = best_sol_cost

    t = 0
    it = 1
    T = int(np.ceil(dim / 10.0) * 60)

    best_cost_hist = []
    globest_hist = []
    curve_fes = []
    curve_best = []

    while fes <= max_fes:
        i0 = rng.integers(0, n, size=m)

        for k in range(m):
            if fes > max_fes:
                break

            i = int(i0[k])
            a = int(i0[k - 1]) if k == m - 1 else int(i0[k + 1])

            X = pop_pos[i].copy()
            X1 = pop_pos[a].copy()
            Xb = best_sol_pos.copy()
            Xbest = x_best_pos.copy()

            kk = 0.0
            if use_improvement_rule:
                lag = int(np.ceil(0.2 * T + 1))
                if (
                    i <= 1  # equivalente a i <= 2 en 1-based MATLAB
                    and t > 2
                    and t > lag
                    and len(best_cost_hist) > lag
                    and len(globest_hist) >= t - 1
                    and abs(best_cost_hist[t - 3] - best_cost_hist[t - lag - 1])
                    <= 0.0001 * abs(globest_hist[t - 2])
                ):
                    X = x_best_pos.copy()
                    kk = 0.0
                elif i == 2:  # equivalente a i == 3 en 1-based MATLAB
                    X = best_sol_pos.copy()
                    kk = -0.1 * rng.random() * (t / max(T, 1))
                else:
                    kk = 0.25

            Z = X.copy()

            r_hat = rng.standard_normal(dim)
            r1 = rng.random(dim)

            if k == 0:  # Líder (k == 1 en MATLAB)
                alpha = alpha_scale * (t / max(T, 1)) * (ub_arr - lb_arr)
            else:  # Miembros del grupo
                step_jitter = 0.001 * np.round((rng.random(dim) > 0.6).astype(float))
                alpha = (
                    alpha_scale * (t / max(T, 1)) * np.abs(Xb - X) + step_jitter
                )

            r = rng.standard_normal(dim)
            r_check = r_check_scale * (
                (np.abs(r) ** np.exp(r / 2.0)) * np.sin(2.0 * np.pi * r)
            )
            beta = X1 - X

            h0 = np.exp(2.0 - 2.0 * (t / max(T, 1)))
            H = np.abs(2.0 * r1 * h0 - h0)

            r2 = rng.random(dim)
            r3 = kk + rng.random(dim)
            r4 = 3.0 * rng.random(dim)

            # Búsqueda vs. Ataque
            mask_search = H > r4
            mask_attack = ~mask_search

            # Evitar división exacta por cero si r_hat tiene algún valor nulo
            safe_r_hat = np.where(np.abs(r_hat) < 1e-15, 1e-15, r_hat)
            Z[mask_search] = X[mask_search] + (safe_r_hat[mask_search] ** -1) * alpha[
                mask_search
            ]
            Z[mask_attack] = Xbest[mask_attack] + r_check[mask_attack] * beta[
                mask_attack
            ]

            # Acecho / espera (Sit & wait)
            mask_sit_wait = r2 > r3
            Z[mask_sit_wait] = X[mask_sit_wait]

            # Manejo de límites (reinicio aleatorio dentro de [lb, ub])
            below = Z < lb_arr
            if np.any(below):
                Z[below] = lb_arr[below] + rng.random(np.sum(below)) * (
                    ub_arr[below] - lb_arr[below]
                )

            above = Z > ub_arr
            if np.any(above):
                Z[above] = lb_arr[above] + rng.random(np.sum(above)) * (
                    ub_arr[above] - lb_arr[above]
                )

            new_cost = float(fobj(Z))
            fes += 1

            if new_cost < pop_cost[i]:
                pop_pos[i] = Z
                pop_cost[i] = new_cost
                if new_cost < best_sol_cost:
                    best_sol_pos = Z.copy()
                    best_sol_cost = new_cost

            curve_fes.append(fes)
            curve_best.append(min(x_best_cost, best_sol_cost))

        t += 1

        # Regresar a casa y cambiar posición del líder si hay estancamiento
        lag_t = int(round(T))
        if t > T and (t - lag_t - 1) >= 1 and t > 2 and len(best_cost_hist) > lag_t:
            if abs(best_cost_hist[t - 2] - best_cost_hist[t - lag_t - 2]) <= abs(
                0.01 * best_cost_hist[t - 2]
            ):
                best_new = x_best_pos.copy()
                num_j = max(1, int(np.ceil((dim / 10.0) * rng.random())))
                j0 = rng.integers(0, dim, size=num_j)
                best_new[j0] = lb_arr[j0] + rng.random(num_j) * (
                    ub_arr[j0] - lb_arr[j0]
                )

                if fes <= max_fes:
                    best_sol_cost = float(fobj(best_new))
                    best_sol_pos = best_new
                    fes += 1

                perm_idx = rng.integers(0, n, size=n)
                if m < n:
                    pop_pos[perm_idx[n - m : n]] = home_pos[perm_idx[:m]]
                pop_pos[i] = x_best_pos.copy()
                pop_cost[i] = x_best_cost

                t = 1

        it += 1

        if best_sol_cost < x_best_cost:
            x_best_pos = best_sol_pos.copy()
            x_best_cost = best_sol_cost

        if len(best_cost_hist) < t:
            best_cost_hist.extend([0.0] * (t - len(best_cost_hist)))
            globest_hist.extend([0.0] * (t - len(globest_hist)))

        best_cost_hist[t - 1] = best_sol_cost
        globest_hist[t - 1] = x_best_cost

    return {
        "best_pos": x_best_pos,
        "best_cost": x_best_cost,
        "fes": fes,
        "iterations": it - 1,
        "curve_fes": np.array(curve_fes),
        "curve_best": np.array(curve_best),
    }