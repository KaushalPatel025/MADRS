# MADRS — Modified Acceleration-Displacement Response Spectrum

A Python package implementing the **MADRS** method of FEMA 440 for seismic performance-point calculation using pushover curves and response spectra.

---

## Installation

### From PyPI (once published)
```bash
pip install madrs
```

### Directly from GitHub
```bash
pip install git+https://github.com/YOUR_USERNAME/MADRS.git
```

### Local development install (editable)
```bash
git clone https://github.com/YOUR_USERNAME/MADRS.git
cd MADRS
pip install -e .
```

---

## Quick Start

```python
import numpy as np
from madrs import MADRS_Method

# --- Pushover curve: columns = [displacement (m), base shear (kN)] ---
PO = np.loadtxt("pushover.csv", delimiter=",")

# --- Demand curve: columns = [period (s), Sa (g)] ---
DC = np.loadtxt("spectrum.csv", delimiter=",")

# --- Modal / structural parameters ---
pf1       = 1.30    # Modal participation factor (1st mode)
alpha1    = 0.85    # Modal mass coefficient (1st mode)
wt        = 5000.0  # Seismic weight (kN)
phi_roof1 = 1.0     # 1st-mode shape at roof

tol  = 1e-4   # Area-balance tolerance for bilinear fitting
CP1  = 0.5    # Lower ay search bound (fraction of api)
CP2  = 0.95   # Upper ay search bound (fraction of api)
CP3 = 0.5     # Controls initial stiffness of the bi-linear curve (ranges from 0 to 1, preferred 0.5)

results = MADRS_Method(PO, DC, pf1, alpha1, wt, phi_roof1, tol, CP1, CP2, CP3)

dpi, api, dy, ay, roof_disp, flag, *_ = results

if flag:
    print(f"Performance point — Sd: {dpi:.4f} m  |  Sa: {api:.4f} g")
    print(f"Roof displacement : {roof_disp:.4f} m")
else:
    print("No performance point found — adjust CP1, CP2, or tol.")
```

---

## API Reference

### `MADRS_Method`

```python
MADRS_Method(PO, DC, pf1, alpha1, wt, phi_roof1, tol, CP1, CP2,
             show_intermediate_plots=True)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `PO` | `ndarray (N, 2)` | Pushover curve — `[displacement (m), base shear (kN)]` |
| `DC` | `ndarray (M, 2)` | Demand/response spectrum — `[period (s), Sa (g)]` |
| `pf1` | `float` | Modal participation factor (1st mode) |
| `alpha1` | `float` | Modal mass coefficient (1st mode) |
| `wt` | `float` | Seismic weight (kN) |
| `phi_roof1` | `float` | 1st-mode shape value at roof |
| `tol` | `float` | Tolerance for bilinear area balance |
| `CP1` | `float` | Lower multiplier for `ay` search |
| `CP2` | `float` | Upper multiplier for `ay` search |
| `CP3` | `float` | Controls initial stiffness of the bi-linear curve (ranges from 0 to 1, preferred 0.5) |
| `show_intermediate_plots` | `bool` | Show 6-panel diagnostic figure (default `True`) |

**Returns** `(dpi, api, dy, ay, roof_disp, flag, Sd_spectra, Sa_Spectranew, Sd, Sa, x_bilinear, y_bilinear)`

### Helper functions

```python
from madrs import curve_intersections, closest_point_on_curve, area_between_curves
```

- **`curve_intersections(x1, y1, x2, y2)`** — find intersection points between two polylines
- **`closest_point_on_curve(P, x_curve, y_curve)`** — nearest point on a polyline to point P
- **`area_between_curves(x, y1, y2)`** — signed area between two curves via Simpson's rule

---

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.24
- scipy ≥ 1.10
- matplotlib ≥ 3.7

---

## License

MIT
