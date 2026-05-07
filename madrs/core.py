import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

TOL_INTER = 5  # 5% error tolerance at intersection point


def area_between_curves(x_common, y1_common, y2_common):
    """Compute the signed area between two curves using Simpson's rule."""
    return simpson(y1_common - y2_common, x=x_common)


def curve_intersections(x1, y1, x2, y2):
    """
    Find intersection points between two curves defined by (x1, y1) and (x2, y2).

    Returns
    -------
    list of (x, y) tuples at intersection points
    """
    x1, y1, x2, y2 = map(np.asarray, (x1, y1, x2, y2))
    y2_interp = np.interp(x1, x2, y2)
    diff = y1 - y2_interp
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]

    intersections = []
    for idx in sign_changes:
        x_left, x_right = x1[idx], x1[idx + 1]
        y_left, y_right = diff[idx], diff[idx + 1]
        x_inter = x_left - y_left * (x_right - x_left) / (y_right - y_left)
        y_inter = np.interp(x_inter, x1, y1)
        intersections.append((x_inter, y_inter))

    return intersections


def closest_point_on_curve(P, x_curve, y_curve):
    """
    Find the closest point on a polyline curve to a given point P.

    Parameters
    ----------
    P : array-like, shape (2,)
    x_curve, y_curve : array-like

    Returns
    -------
    best_point : ndarray, shape (2,)
    min_dist : float
    percent_error : ndarray, shape (2,)
    """
    P = np.array(P, dtype=float)
    x_curve = np.array(x_curve, dtype=float)
    y_curve = np.array(y_curve, dtype=float)

    min_dist = np.inf
    best_point = None

    for i in range(len(x_curve) - 1):
        A = np.array([x_curve[i], y_curve[i]])
        B = np.array([x_curve[i + 1], y_curve[i + 1]])
        AB = B - A
        AP = P - A
        t = np.clip(np.dot(AP, AB) / np.dot(AB, AB), 0, 1)
        Q = A + t * AB
        dist = np.linalg.norm(P - Q)

        if dist < min_dist:
            min_dist = dist
            best_point = Q

    if best_point is None:
        raise ValueError("No valid closest point found on curve")

    percent_error = np.abs((P - best_point) / (best_point + 1e-12)) * 100
    return best_point, min_dist, percent_error


def MADRS_Method(PO, DC, pf1, alpha1, wt, phi_roof1, tol, CP1, CP2, show_intermediate_plots=True):
    """
    Modified Acceleration-Displacement Response Spectrum (MADRS) method.

    Parameters
    ----------
    PO : ndarray, shape (N, 2)
        Pushover / capacity curve. Columns: [displacement (m), base shear (kN)].
    DC : ndarray, shape (M, 2)
        Demand curve (elastic response spectrum). Columns: [period (s), Sa (g)].
    pf1 : float
        Modal participation factor for the first mode.
    alpha1 : float
        Modal mass coefficient for the first mode.
    wt : float
        Seismic weight of the structure (kN).
    phi_roof1 : float
        First-mode shape value at roof level.
    tol : float
        Tolerance for area difference in bilinear curve fitting.
    CP1 : float
        Lower bound multiplier for ay search (fraction of api).
    CP2 : float
        Upper bound multiplier for ay search (fraction of api).
    show_intermediate_plots : bool, optional
        If True (default), display the 6-panel diagnostic figure.

    Returns
    -------
    dpi : float or None
        Spectral displacement at performance point (m).
    api : float or None
        Spectral acceleration at performance point (g).
    dy : float or None
        Yield spectral displacement of bilinear curve (m).
    ay : float or None
        Yield spectral acceleration of bilinear curve (g).
    roof_disp : float or None
        Roof displacement at performance point (m).
    flag : int
        1 if a performance point was found, 0 otherwise.
    Sd_spectra : ndarray or None
        Spectral displacements of the demand spectrum.
    Sa_Spectranew : ndarray or None
        Scaled demand spectrum spectral accelerations.
    Sd : ndarray or None
        Capacity spectrum spectral displacements.
    Sa : ndarray or None
        Capacity spectrum spectral accelerations.
    x_bilinear : ndarray or None
        x-coordinates of fitted bilinear curve.
    y_bilinear : ndarray or None
        y-coordinates of fitted bilinear curve.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    # --- Capacity curve ---
    pushover_curve = PO
    axes[0].plot(pushover_curve[:, 0], pushover_curve[:, 1],
                 label="Capacity Curve", color="blue")
    axes[0].set_xlabel("Displacement (m)", fontsize=12)
    axes[0].set_ylabel("Base Shear (kN)", fontsize=12)
    axes[0].legend(loc="lower right", framealpha=0.0, fontsize=10)

    # --- Truncate at 0.8 * RF_max ---
    rf_max = np.max(pushover_curve[:, 1])
    index = np.argmax(pushover_curve[:, 1])
    narray = pushover_curve[index:, 1]
    narray2 = narray - 0.8 * rf_max
    index_min = np.argmin(np.abs(narray2))
    po_y = pushover_curve[: index + index_min, 1]
    po_x = pushover_curve[: index + index_min, 0]

    axes[1].plot(po_x, po_y, label="Capacity Curve up to 0.8*Max", color="green")
    axes[1].set_xlabel("Displacement (m)", fontsize=12)
    axes[1].set_ylabel("Base Shear (kN)", fontsize=12)
    axes[1].legend(loc="lower right", framealpha=0.0, fontsize=10)

    # --- Capacity spectrum (Sa, Sd) ---
    Sa = po_y / (wt * alpha1)
    Sd = po_x / (pf1 * phi_roof1)
    axes[2].plot(Sd, Sa, label="Capacity Spectrum", color="blue")
    axes[2].set_xlabel("Spectral Displacement (m)", fontsize=12)
    axes[2].set_ylabel("Spectral Acceleration (g)", fontsize=12)
    axes[2].legend(loc="lower right", framealpha=0.0, fontsize=10)

    # --- Demand curve ---
    EC_Spectrum = DC
    axes[3].plot(EC_Spectrum[:, 0], EC_Spectrum[:, 1],
                 label="Demand Curve", color="red")
    axes[3].set_xlabel("Time Period (Sec)", fontsize=12)
    axes[3].set_ylabel("Spectral Acceleration (g)", fontsize=12)
    axes[3].legend(loc="lower right", framealpha=0.0, fontsize=10)

    # --- Demand spectrum in Sa-Sd space ---
    Sd_spectra = EC_Spectrum[:, 1] * (EC_Spectrum[:, 0] ** 2) / (4 * np.pi ** 2)
    axes[4].plot(Sd_spectra, EC_Spectrum[:, 1],
                 label="Demand Spectrum", color="orange")
    axes[4].set_xlabel("Spectral Displacement (m)", fontsize=12)
    axes[4].set_ylabel("Spectral Acceleration (g)", fontsize=12)
    axes[4].legend(loc="lower right", framealpha=0.0, fontsize=10)

    # --- Initial slope & initial guess ---
    Sa_per_ind = np.argmin(np.abs(Sa - 0.6 * Sa.max()))
    k_init = (Sa[Sa_per_ind] - Sa[0]) / (Sd[Sa_per_ind] - Sd[0])
    x2 = np.array([0, 1 / k_init])
    y2 = np.array([0, 1])

    points = curve_intersections(Sd_spectra, EC_Spectrum[:, 1], x2, y2)
    print("Intersection points:", points)

    x3 = np.array([points[0][0], points[0][0]])
    y3 = np.array([0, points[0][1]])
    points2 = curve_intersections(Sd, Sa, x3, y3)
    x4 = points2[-1][0]
    y4 = points2[-1][1]

    axes[5].plot(Sd, Sa, label="Capacity Spectrum", color="blue")
    axes[5].plot(Sd_spectra, EC_Spectrum[:, 1],
                 label="Demand Spectrum", color="orange")
    axes[5].plot(x2, y2, label="Initial Slope")
    axes[5].plot(points[0][0], points[0][1], "ro", label="Intersection")
    axes[5].plot(x4, y4, "go", label="Initial Guess")
    axes[5].set_xlabel("Spectral Displacement (m)", fontsize=12)
    axes[5].set_ylabel("Spectral Acceleration (g)", fontsize=12)
    axes[5].legend(loc="upper right", framealpha=0.0, fontsize=10)

    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()

    if show_intermediate_plots:
        plt.show()
    else:
        plt.close(fig)

    # --- Main MADRS iteration ---
    loc = np.argmin(abs(Sa - points2[-1][1]))

    for newloc in np.arange(int(0.1 * loc), len(Sa), 2):
        x4 = Sd[newloc]
        y4 = Sa[newloc]

        for i in np.arange(CP1 * y4, CP2 * y4, (CP2 - CP1) * y4 / 100):
            x = np.array([0])
            y = np.array([0])
            y = np.append(y, i)
            x = np.append(x, y[-1] / k_init)
            x = np.append(x, x4)
            y = np.append(y, y4)

            x_common = np.unique(np.concatenate([x, Sd[:newloc]]))
            x_common = x_common[x_common <= x[-1]]
            Sa_common = np.interp(x_common, Sd, Sa)
            y_common = np.interp(x_common, x, y)
            A_diff = area_between_curves(x_common, Sa_common, y_common)

            if abs(A_diff) < tol:
                break

        api = y[-1]
        dpi = x[-1]
        ay = y[1]
        dy = x[1]
        mue = dpi / dy

        beta_zero = 5
        T_zero = 2 * np.pi * (dy / ay) ** 0.5

        if mue < 4:
            T_eff = (0.2 * (mue - 1) ** 2 - 0.038 * (mue - 1) ** 3 + 1) * T_zero
        elif 4 <= mue <= 6.5:
            T_eff = (0.28 + 0.13 * (mue - 1) + 1) * T_zero
        else:
            T_eff = (0.89 * (((mue - 1) / (1 + 0.05 * (mue - 2))) ** 0.5 - 1) + 1) * T_zero

        if mue < 4:
            beta_eff = 4.9 * (mue - 1) ** 2 - 1.1 * (mue - 1) ** 3 + beta_zero
        elif 4 <= mue <= 6.5:
            beta_eff = 14 + 0.32 * (mue - 1) + beta_zero
        else:
            beta_eff = (
                19
                * ((0.64 * (mue - 1) - 1) / ((0.64 * (mue - 1)) ** 2))
                * ((T_eff / T_zero) ** 2)
                + beta_zero
            )

        alpha = ((api - ay) / (dpi - dy)) / (ay / dy)
        B_beta_eff = 4 / (5.6 - np.log(beta_eff))
        T_sec = T_zero / ((1 + alpha * (mue - 1)) / mue) ** 0.5
        M = (T_eff / T_sec) ** 2

        Sa_Spectranew = (EC_Spectrum[:, 1] / B_beta_eff) * M

        P = [x[-1], y[-1]]
        Q, error, percent_error = closest_point_on_curve(P, Sd_spectra, Sa_Spectranew)

        if percent_error[0] <= TOL_INTER and percent_error[1] <= TOL_INTER:
            flag = 1

            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.plot(Sd_spectra, EC_Spectrum[:, 1],
                     label="Demand Spectrum (EC2)", color="orange")
            ax2.plot(Sd_spectra, Sa_Spectranew,
                     label="Demand Spectrum (Scaled)", color="green")
            ax2.plot(Sd, Sa, label="Capacity Spectrum", color="blue")
            ax2.plot(x, y, label="Bilinear Curve", color="cyan")
            ax2.plot(x[-1], y[-1], "ro", label="api, dpi")
            ax2.plot(x[1], y[1], "go", label="ay, dy")
            ax2.legend()
            ax2.set_title("MADRS Approach")
            ax2.set_xlim(left=0)
            ax2.set_ylim(bottom=0)
            ax2.set_xlabel("Spectral displacement (m)")
            ax2.set_ylabel("Spectral acceleration (g)")
            plt.tight_layout()
            plt.show()

            print("Passed")
            print("Area difference (curve1 - curve2) =", A_diff)

            BOLD_RED = "\033[1;31m"
            RESET = "\033[0m"
            if abs(A_diff) > tol:
                print(f"{BOLD_RED}Area difference is more than tolerance, "
                      f"change bi-linear curve style{RESET}")

            print("Mue =", mue)
            print("Effective Damping =", beta_eff)
            roof_disp = x[-1] * (pf1 * phi_roof1)
            print("Roof displacement at performance point (m) =", roof_disp)

            return dpi, api, dy, ay, roof_disp, flag, Sd_spectra, Sa_Spectranew, Sd, Sa, x, y

        else:
            flag = 0

    print("No performance point found. Try adjusting CP1, CP2, or tol.")
    return None, None, None, None, None, 0, None, None, None, None, None, None
