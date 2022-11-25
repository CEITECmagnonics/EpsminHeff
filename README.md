# EpsminHeff
Computation of energy density minima and effective field values.

## Requirements
I'm not sure about all dependencies, but here are package versions I used:
Python 3.10.8
Numpy 1.23.5
Matplotlib 3.6.2
Scipy 1.9.3
myfxs (optional, see [MyFxs Package](https://github.com/GiovanniKl/MyFxsPackage)

The Myfxs package turns all fonts into serif fonts. You can omit this package, if you always set the `serif` parameter to `False`. *(Note: This might change in the future.)*

## Usage

This package calculates energy density $\epsilon$ and effective magnetic field $\textbf{H}_{\mathrm{eff}}$ of a rectangle or a cylinder homogenously magnetized (to saturation) in the $xy$ plane. These are functions of magnetization direction characterized by the angle $\varphi$. You can choose what components you want to include:
- dipolar (demagnetization) energy density ${\epsilon_{\mathrm{dip}}}$ and field ${\textbf{H}_{\mathrm{dip}}}$
- uniaxial anisotropy energy density ${\epsilon_{\mathrm{ani}}}$ and field ${\textbf{H}_{\mathrm{ani}}}$
- Zeeman energy density ${\epsilon_{\mathrm{Zee}}}$ and external field ${\textbf{H}_{\mathrm{ext}}}$.

You can find and plot the total energy density minimum ${\epsilon_{\mathrm{min}}}$, the corresponding angle $\varphi_0$ and the value of the effective field ${H_{\mathrm{eff}}}$ for ${\epsilon=\epsilon_{\mathrm{min}}}$.

All magnetic fields are used in units of magnetic induction (tesla), that is, $\mu_0 H$.

You can do only calculations or also plot the results of energy density in polar or rectilinear projection (for negative values rectilinear projection is preferred). 

![polar example](/test_plots/ani20221125_00_test_eden_polar.png)

![rectilinear example](/test_plots/ani20221125_00_test_eden_rect.png)

You can also plot the effective field and its components. Here the color signalizes the angle $\varphi$ continuously changing from $0$ to $2\pi$ (color light to dark).

![heff example](/test_plots/ani20221125_00_test_heff.png)

If you are interested, you can also plot the magnetic induction but, in case of weak fields, this plot is quite useless.

![beff example](/test_plots/ani20221125_00_test_beff.png)

For complete examples see `main()` functions in `epsmin_heff.py` and `cylinder_test.py`.


## Reference
This reference guide might not be complete. For better understanding of how things work check the script inside the python files.

### `epsmin_heff.py`

Main module file.

### Available constants

#### `MU0`
Magnetic permeability of vacuum in SI units computed as `4e-7*np.pi`.

### Available functions

#### `main()`
Function used for direct execution of the `epsmin_heff.py` file. May serve as an example of usage for this module.

#### `ytilt(theta)`
Function that returns absolute angle [°] from $y$ axis, e.g. `ytilt(60*np.pi/180) == 30 == ytilt(120*np.pi/180)`. 
- `theta` - float, [rad] positive angle from x axis.

#### `ku2bani(ku, msat)`
Function that recalculates uniaxial anisotropy constant to equivalent anisotropic magnetic field.
- `ku` - [J/m3] uniaxial anisotropy constant $K_{\mathrm{u}}$.
- `msat` - [A/m] saturation magnetisation $M_{\mathrm{s}}$.
Returns:
- `bani` - [T] anisotropic magnetic field $\mu_0 H_{\mathrm{ani}}$.

#### `bani2ku(bani, msat)`
Function that recalculates anisotropic magnetic field to equivalent uniaxial anisotropy constant.
- `bani` - [T] anisotropic magnetic field $\mu_0 H_{\mathrm{ani}}$.
- `msat` - [A/m] saturation magnetisation $M_{\mathrm{s}}$.
Returns:
- `ku` - [J/m3] uniaxial anisotropy constant $K_{\mathrm{u}}$.

#### `aharoni(a, b, c)`
Returns demagnetization factor Nz from Aharoni model. To get Nx and Ny use (twice) the cyclic permutation $c\to a\to b\to c$.
- `a`, `b`, `c` - float, [m] prism overall dimensions in the $xyz$ order.

#### `wysin_cylinder(h, r)`
Returns diagonal components $N_x$, $N_y$, $N_z$ of the demagnetization tensor from Wysin's solution for a homogenously magnetized cylinder with rotational axis along z drection.  For more info click [here](https://www.phys.ksu.edu/personal/wysin/notes/demag.pdf).
- `h` - float or ndarray, [m] height of the cylinder.
- `r` - float or ndarray of the same shape as `h`, [m] radius of the cylinder's circular base.

### Available classes

#### `EpsminHeff((self, msat, name, loc="", dpi=250, npoints=100, use_dip=False, use_uniax=False, use_bext=False, plot_total=False, fit_angle=True, plot_other_angles=True, plot_rectilinear=True, plot_polar=False, plot_heff=True, plot_beff=True, save_pdf=False, serif=True, title=None, save_metadata=True, demag_factors=(0., 0., 1.), ku=0., delta=0., bext=0., xi=0.)`

Class characterizing the process of finding the minima of the energy density $\epsilon_{\mathrm{tot}}$ and the value of the effective field $\mu_0 \textbf{H}_{\mathrm{eff}}$ in the $xy$ plane.

**Keyword Args:**
- `msat` - float, [A/m] saturation magnetization of the magnetic body $M_{\mathrm{s}}$.
- `name` - str, common name of the plots and metadata files (without extension).
- `loc` - str (default `""`), base directory for saving files into. Preferably, use slashes instead of backslashes.
- `dpi` - int (default `250`), DPI resolution for saved plots.
- `npoints` - int (default `100`), number of calculation nodes for both, energy density and effective field.
- `use_dip` - bool (default `False`), whether to account for dipolar (demagnetizing) field/energy density.
- `use_uniax` - bool (default `False`), whether to account for uniaxial anisotropy field/energy density.
- `use_bext` - bool (default `False`), whether to account for external magnetic field/Zeeman energy density.
- `plot_total` - bool (default `False`), whether to plot the sum of accounted fields/energy densities.
- `fit_angle` - bool (default `False`), whether to fit/find and plot the angle of the total energy density at its minimum and the angle of the total effective magnetic induction at its maximum (or rather at the angle of magnetization corresponding to energy density minimum).
- `plot_other_angles` - bool (default `True`), whether to plot angles of minima of all energy density components (except dipolar energy density).
- `plot_rectilinear` - bool (default `True`), whether to make a plot of energy density in rectilinear projection.
- `plot_polar` - bool (default `True`), whether to make a plot of energy density in polar projection.
- `plot_heff` - bool (default `True`), whether to make a plot of effective magnetic field (and its components).
- `plot_beff` - bool (default `True`), whether to make a plot of effective magnetic induction (and its components).
- `save_pdf` - bool (default `False`), whether to save all plots also in PDF format.
- `serif` - bool (default `True`), whether to plot text in serif font family.
- `title` - `None` or str (default `None`), if given as a str, this will be used as a title for all plots. *Note: To have different titles on your figures, you can change this value with `self.title` between calling each plotting method.*
- `save_metadata` - bool (default `True`), whether to save all computation parameters into a TXT file.
- `demag_factors` - 3-tuple (or list) of floats (default `(0., 0., 1.)`), diagonal components of the demagnetizing tensor. This script accounts only for geometries of the body that have a diagonal demagnetizing tensor. This may change in the future. The default value corresponds to an infinite layer in the computed plane.
- `ku` - float (default `0.`), [J/m^3] uniaxial anisotropy constant $K_{\mathrm{u}}$.
- `delta` - float (default `0.`), [rad] tilt of uniaxial anisotropy axis from the $x$ axis.
- `bext` - float (default `0.`), [T] external magnetic field $\mu_0 H_{\mathrm{ext}}$ (in units of magnetic induction).
- `xi` - float (default `0.`), [rad] tilt of $\mu_0 H_{\mathrm{ext}}$ from the $x$ axis.

**Stored Values:**
(for kwargs info see above)
- `msat` - `msat` init kwarg.
- `name` - `name` init kwarg.
- `loc` - `loc` init kwarg with a slash appended at the end if it was not there before.
- `dpi` - `dpi` init kwarg.
- `n` - `npoints` init kwarg.
- `use` - list of bools, defines used model parts from init kwargs.
- `plot` - list of bools, defines plotting parameters from init kwargs.
- `title` - `title` init kwarg.
- `metadata` - `save_metadata` init kwarg.
- `demfs` - `demag_factors` init kwarg.
- `ku` - `ku` init kwarg.
- `delta` - `delta` init kwarg.
- `bext` - `bext` init kwarg.
- `xi` - `xi` init kwarg.
- `phi` - ndarray of shape `(n,)`, [rad] angles of magnetization to compute at.
- `eden` - ndarray of shape `(4, n)`, [J/m^3] energy density components in this order: dipolar, anisotropy, Zeeman, total.
- `htot` - ndarray of shape `(2, n)`, [T] $x$ and $y$ components of $\mu_0 H_{\mathrm{eff}}$.
- `hdip` - ndarray of shape `(2, n)`, [T] $x$ and $y$ components of $\mu_0 H_{\mathrm{dip}}$.
- `hani` - ndarray of shape `(2, n)`, [T] $x$ and $y$ components of $\mu_0 H_{\mathrm{ani}}$.
- `phi_emin` - `None` or float, [rad] phi at minimal total energy density, also sometimes referred to as $\varphi_0$.
- `emin` - `None` or float, [J/m^3] value of energy density at its minimum.
- `h_emin` - `None` or 2-list of floats, [T] $x$ and $y$ value of effective magnetic field $\mu_0 H_{\mathrm{eff}}$ at energy minimum, or at `phi_emin` respectively, since $\mu_0 H_{\mathrm{eff}}$ and $M$ should be parallel at energy minimum.
- `colors` - 4-list of colormaps for `n` values (default colormaps are `Reds`, `Blues`, `Greens`, `Oranges`), these colormaps are used in $\mu_0 H_{\mathrm{eff}}$ and $B_{\mathrm{eff}}$ plots for fields in this order: dipolar, anisotropy, external, total. By default these colormaps correspond to colors in the `color` (or `self.color` if you prefer).
- `color` - 4-list of color strings (default ["tab:red", "tab:blue", "tab:green", "tab:orange"]), these are used for plotting energy density in this order: dipolar, anisotropy, Zeeman, total (and corresponding effective field components' legend keys).
- `line_width` - float (default `1.5`), line width of energy densities in plots.
- `figs_size` - 4-list of 2-tuples of ints or floats (default `[(4, 3), (4, 3), (6, 4), (6, 4)]`), [inch] each tuple represents figsize parameter for each plot in this order: rectilinear, polar, heff, beff.
- `elabels` - 4-list of strings, legend labels for energy density in the same order as e.g. in `color`.
- `hlabels` - 5-list of strings, legend labels for effective magnetic field ant its components in the same order as e.g. in `colors`, plus for effective field in energy minimum.
- `blabels` - 4-list of strings, legend labels for effective magnetic induction and its components in the same order as e.g. in `colors`.
- `angles` - 4-list of strings, angle names used in plots in this order: `delta`, `xi`, `phi`, `phi_emin`.
- `pe` - list of `path_effects` objects, path effects for advanced formatting of text plotted using `fit_angle` and `plot_other_angles`. By default this creates a 0.5-thick white outline around the letters (for better clarity of drawn angle values).

**Methods:**
- `compute_energy_density(self)` - calculates all the necessary energy density components and its minimum position.
- `compute_effective_field(self)` - calculates all the necessary magnetic field components (as $\mu_0 H$, that is, in units of induction).
- `save_metadata(self)` - saves all calculation parameters and results into a TXT file.
- `plot_rectilinear(self)` - generates a plot of energy density and its components in rectilinear projection.
- `plot_polar(self)` - generates a plot of energy density and its components in polar projection. *Note: This plots only positive values of energy! For expected negative values, rectilinear projection is recommended.*
- `plot_heff(self)` - generates a quiver plot of effective magnetic field and its components. If `fit_angle` is `True`, this also plots effective magnetic field for angle `phi_emin` (angle of magnetization from $x$ axis at energy density minimum).
- `plot_beff(self)` - generates a quiver plot of effective magnetic induction B_eff = mu_0(H_eff + M) (MathJax did not cooperate here) and its components. If `fit_angle` is `True`, also plots effective magnetic induction for angle `phi_emin` (angle of magnetization from $x$ axis at energy density minimum).
- `evaluate(self)` - automatic processiing of the calculation and plotting according to the calculation setup.


### `cylinder_test.py`

This file serves as an example for using the `epsmin_heff.py` module. It uses the cylindrical demagnetization factors to show the difference between Aharoni and Wysin models.



