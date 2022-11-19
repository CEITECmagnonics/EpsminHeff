"""
eps_min and H_eff calculation script.  Can be used as a module.
Useful in magnonics research.

Included classes:
    EpsminHeff - Compute energy density minima and effective field
        value in the xy plane.

Included functions:
     main - used for direct execution of this file.  May serve as an
        example of usage for this module.
     ytilt - from deviation from x axis in rad returns deviation from
        y axis in °.
    ku2bani - converts uniaxial anisotropy constant to equivalent
        anisotropic magnetic field.
    bani2ku - converts anisotropic magnetic field to equivalent
        uniaxial anisotropy constant.
    aharoni - calculates demagnetization factors of a rectangular
        prism using the Aharoni model.

@author: Jan Klíma, jan.klima4@vutbr.cz
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import myfxs as mf  # see https://github.com/GiovanniKl/MyFxsPackage

MU0 = 4*np.pi*1e-7  # [N/A^2] permeability of vacuum


def main():
    """Function setting all the parameters and controlling the
    computation.  The same could be called from outside."""


class EpsminHeff:
    """Class characterizing the process of finding the minima of
     the energy density eps_tot and value of the effective field
     mu_0*H_eff in the xy plane.
     Keyword Args:
         msat - float, [A/m] saturation magnetization of the magnetic
            body M_sat.
        name - str, common name of the plot and metadata files
            (without extension).
        loc - str (default ""), base directory for saving files into.
        dpi - int (default 250), DPI resolution for saved plots.
        npoints - int (default 100), number of calculation nodes for
            both, energy density and effective field.
        use_dem - bool (default False), whether to account for
            demagnetizing field/energy density.
        use_uniax - bool (default False), whether to account for
            uniaxial anisotropy field/energy density.
        use_bext - bool (default False), whether to account for
            external magnetic field/Zeeman energy density.
        plot_total - bool (default False), whether to plot the sum of
            accounted fields/energy densities.
        fit_angle - bool (default False), whether to fit/find and plot
            the angle of the total energy density at it's minimum and
            the angle of the total effective magnetic induction at
            it's maximum.
        add_title - bool (default True), whether to add a descriptive
            title to all plots.
        plot_heff - bool (default True), whether to make a plot of
            effective magnetic field (and it's components).
        plot_beff - bool (default True), whether to make a plot of
            effective magnetic induction (and it's components).
        save_pdf - bool (default False), whether to save all plots
            also in PDF format.
        save_metadata - bool (default True), whether to save all
            computation parameters into a TXT file.
        demag_factors - 3 tuple (or list) of floats
            (default (0, 0, 1)), diagonal components of the
            demagnetizing tensor. This script accounts only for
            geometries of the body that have a diagonal demagnetizing
            tensor. This may change in the future. The default value
            corresponds to an infinite layer in the computed plane.
        ku - float (default 0), [J/m^3] uniaxial anisotropy
            constant K_u.
        tilt_uni - float (default 0), [rad] tilt of uniaxial anisotropy
            axis from the x axis.
        bext - float (default 0), [T] external magnetic field
            mu_0*H_ext (in units of magnetic induction).
        ksi - float (default 0), [rad] tilt of H_ext from the x axis.

    Methods:

    """
    def __init__(self, msat, name, loc="", dpi=250, npoints=100,
                 use_dem=False, use_uniax=False, use_bext=False,
                 plot_total=False, fit_angle=True, add_title=True,
                 plot_heff=True, plot_beff=True, save_pdf=False,
                 save_metadata=True,
                 demag_factors=(0, 0, 1),
                 ku=0, tilt_uni=0,
                 bext=0, ksi=0):
        self.msat = msat
        self.name = name
        self.loc = loc
        self.dpi = dpi
        self.n = npoints
        self.use = [use_dem, use_uniax, use_bext]
        self.plot = [plot_total, fit_angle, add_title, plot_heff, plot_beff,
                     save_pdf]
        self.metadata = save_metadata
        self.demfs = demag_factors
        self.ku = ku
        self.tiltuni = tilt_uni
        self.bext = bext
        self.ksi = ksi
        # compute phi vector - [rad] angles of magnetization to compute at
        self.phi = np.linspace(0, 2*np.pi, self.n)
        # preallocation for other used variables
        self.etot, self.edip, self.eani, self.ezez = (None,)*4
        self.tilttot = 0

    def compute_energy_density(self):
        """Computes all energy density components."""
        self.etot = np.zeros(self.n)
        if self.use[0]:
            self.edip = (MU0/2 * self.msat**2
                         * (self.demfs[0] * np.cos(self.phi)**2
                            + self.demfs[1] * np.sin(self.phi)**2))
            self.etot += self.edip
        # CONTINUE HERE


def ytilt(theta):
    """Function that returns absolute angle [°] from y axis, e.g.
    ytilt(60*np.pi/180) == 30 == ytilt(120*np.pi/180).
    theta - float, [rad] positive angle from x axis.
    """
    return np.abs(theta*180/np.pi-90)


def ku2bani(ku, msat):
    """Function that recalculates uniaxial anisotropy constant to
    equivalent anisotropic magnetic field.
    ku - [J/m3] anisotropy constant.
    msat - [A/m] saturation magnetisation.
    Returns:
    bani - [T] anisotropic magnetic field mu_0*H_ani.
    """
    return 2*ku/msat


def bani2ku(bani, msat):
    """Function that recalculates anisotropic magnetic field to
    equivalent uniaxial anisotropy constant.
    bani - [T] anisotropic magnetic field mu_0*H_ani.
    msat - [A/m] saturation magnetisation.
    Returns:
    ku - [J/m3] anisotropy constant.
    """
    return msat*bani/2


def aharoni(a, b, c):
    """Returns demagnetization factor Nz from Aharoni model.
    To get Nx and Ny use (twice) the cyclic permutation c->a->b->c.
    a, b, c - float, [m] prism overall dimensions."""
    a, b, c = a/2, b/2, c/2
    r = np.sqrt(a**2+b**2+c**2)
    ab, ac, bc = np.sqrt(a**2+b**2), np.sqrt(a**2+c**2), np.sqrt(b**2+c**2)
    return ((b**2-c**2)/2/b/c*np.log((r-a)/(r+a)) +
            (a**2-c**2)/2/a/c*np.log((r-b)/(r+b)) +
            b/2/c*np.log((ab+a)/(ab-a)) + a/2/c*np.log((ab+b)/(ab-b)) +
            c/2/a*np.log((bc-b)/(bc+b)) + c/2/b*np.log((ac-a)/(ac+a)) +
            2*np.arctan(a*b/c/r) + (a**3+b**3-2*c**3)/3/a/b/c +
            c/a/b*(ac+bc) + (a**2+b**2-2*c**2)*r/3/a/b/c -
            (ab**3+bc**3+ac**3)/3/a/b/c)/np.pi


if __name__ == "__main__":
    main()
