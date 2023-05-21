"""
eps_min and H_eff calculation script.  Can be used as a module.
Useful in magnonics research.

Included classes:
    EpsminHeff - Compute energy density minima and effective field
        value in the xy plane.
    Hysteresis - Compute hysteresis loops of Stoner-Wohlfarth-like
        origin based on EpsminHeff class. Restriction to xy plane
        still applies.

Included functions:
    main - used for direct execution of this file.  May serve as an
        example of usage for this module.
    printif - print text in terminal upon a condition.
    ytilt - from deviation from x axis in rad returns deviation from
        y axis in °.
    ku2bani - converts uniaxial anisotropy constant to equivalent
        anisotropic magnetic field.
    bani2ku - converts anisotropic magnetic field to equivalent
        uniaxial anisotropy constant.
    aharoni - calculates demagnetization factors of a homogenously
        magnetized rectangular prism using the Aharoni model.
    wysin_cylinder - calculates demagnetization factors of
        a homogenously magnetized cylinder.

@author: Jan Klíma, jan.klima4@vutbr.cz
"""

import numpy as np  # for vectorization
import matplotlib.pyplot as plt  # for plotting
import matplotlib.patheffects as mpe  # for text formatting in plots
from scipy.optimize import fmin, fminbound  # for energy minima finding
from scipy.interpolate import interp1d  # for smaller discretization effects
from scipy.signal import find_peaks  # for energy peak finding (in Hysteresis)

MU0 = 4*np.pi*1e-7  # [N/A^2] permeability of vacuum


def main():
    """Function for setting all the parameters and controlling the
    computation.  The same could be called from outside."""
    name, dpi = "ani20221125_00_test", 250
    loc = "test_plots"
    n = 100
    # rectangular prism -> demag factors from Aharoni model
    dims = (25e-6, 5e-6, 10e-9)  # [m] dimensions of the body in xyz order
    demfs = (aharoni(dims[1], dims[2], dims[0]),
             aharoni(dims[2], dims[0], dims[1]), aharoni(*dims))
    msat = 830e3  # [A/m] saturation magnetization M_s
    ku = bani2ku(20e-3, msat)  # [J/m^3] uniaxial anisotropy constant K_u
    delta = (90-20)*np.pi/180  # [rad] tilt of uniax. anisotropy from x axis
    bext = 5e-3  # [T] external magnetic inducton B_ext
    xi = (90 + 50) * np.pi / 180  # [rad] tilt of B_ext from x axis
    title = "Testing title for some composition of the plot."
    darling = EpsminHeff(msat, name, loc, dpi, n, True, True, True, True, True,
                         True, True, True, True, True, False, title=title,
                         demag_factors=demfs, ku=ku, delta=delta, bext=bext,
                         xi=xi)
    # custom change of angle names from varphi to alpha
    darling.angles[2], darling.angles[3] = r"$\alpha$", r"$\alpha_0$"
    darling.evaluate()
    darling.title = "New custom title."
    darling.name = name+"_newtitle"  # new name not to overwrite the old plot
    darling.plot_heff()  # plot a new graph with new title


class EpsminHeff:
    """Class characterizing the process of finding the minima of
     the energy density eps_tot and the value of the effective field
     mu_0*H_eff in the xy plane.

     Keyword Args:
         msat - float, [A/m] saturation magnetization of the magnetic
            body M_sat.
        name - str, common name of the plots and metadata files
            (without extension).
        loc - str (default ""), base directory for saving files into.
            Preferably, use slashes instead of backslashes.
        dpi - int (default 250), DPI resolution for saved plots.
        npoints - int (default 100), number of calculation nodes for
            both, energy density and effective field.
        use_dip - bool (default False), whether to account for dipolar
            (demagnetizing) field/energy density.
        use_uniax - bool (default False), whether to account for
            uniaxial anisotropy field/energy density.
        use_bext - bool (default False), whether to account for
            external magnetic field/Zeeman energy density.
        plot_total - bool (default False), whether to plot the sum of
            accounted fields/energy densities.
        fit_angle - bool (default False), whether to fit/find and plot
            the angle of the total energy density at its minimum and
            the angle of the total effective magnetic induction at
            its maximum (or rather at the angle of magnetization
            corresponding to energy density minimum).
        plot_other_angles - bool (default True), whether to plot
            angles of minima of all energy density components
            (except dipolar energy density).
        plot_rectilinear - bool (default True), whether to make a plot
            of energy density in rectilinear projection.
        plot_polar - bool (default True), whether to make a plot
            of energy density in polar projection.
        plot_heff - bool (default True), whether to make a plot of
            effective magnetic field (and its components).
        plot_beff - bool (default True), whether to make a plot of
            effective magnetic induction (and its components).
        save_pdf - bool (default False), whether to save all plots
            also in PDF format.
        title - None or str (default None), if given as a str, this
            will be used as a title for all plots.  Note: To have
            different titles on your figures, you can change this
            value with self.title between calling each plotting method.
        save_metadata - bool (default True), whether to save all
            computation parameters into a TXT file.
        demag_factors - 3-tuple (or list) of floats
            (default (0., 0., 1.)), diagonal components of the
            demagnetizing tensor.  This script accounts only for
            geometries of the body that have a diagonal demagnetizing
            tensor.  This may change in the future. The default value
            corresponds to an infinite layer in the computed plane.
        ku - float (default 0.), [J/m^3] uniaxial anisotropy
            constant K_u.
        delta - float (default 0.), [rad] tilt of uniaxial anisotropy
            axis from the x axis.
        bext - float (default 0.), [T] external magnetic field
            mu_0*H_ext (in units of magnetic induction).
        xi - float (default 0.), [rad] tilt of H_ext from the x axis.
        disp_messages - bool (default True), whether to print status
            messages into terminal.

    Stored Values:
        (for kwargs info see above)
        msat - msat init kwarg.
        name - name init kwarg.
        loc - loc init kwarg with a slash appended at the end if
            it was not there before and if the loc kwarg is not "".
        dpi - dpi init kwarg.
        n - npoints init kwarg.
        use - list of bools, defines used model parts from init
            kwargs.
        plot - list of bools, defines plotting parameters from init
            kwargs.
        title - title init kwarg.
        metadata - save_metadata init kwarg.
        demfs - demag_factors init kwarg.
        ku - ku init kwarg.
        delta - delta init kwarg.
        bext - bext init kwarg.
        xi - xi init kwarg.
        mess - disp_messages init kwarg.
        phi - ndarray of shape (n,), [rad] angles of magnetization to
            compute at.
        eden - ndarray of shape (4, n), [J/m^3] energy density
            components in this order: dipolar, anisotropy, Zeeman,
            total.
        htot - ndarray of shape (2, n), [T] x and y components of
            mu_0*H_eff.
        hdip - ndarray of shape (2, n), [T] x and y components of
            mu_0*H_dip.
        hani - ndarray of shape (2, n), [T] x and y components of
            mu_0*H_ani.
        phi_emin - None or float, [rad] phi at minimal total energy
            density, also sometimes referred to as phi_0.
        emin - None or float, [J/m^3] value of energy density at its
            minimum.
        h_emin - None or 2-list of floats, [T] x and y value of
            effective magnetic field mu_0*H_eff at energy minimum,
            or at phi_emin respectively, since H_eff and M should be
            parallel at energy minimum.
        colors - 4-list of colormaps for n values (default colormaps
            are Reds, Blues, Greens, Oranges), these colormaps are
            used in H_eff and B_eff plots for fields in this order:
            dipolar, anisotropy, external, total.  By default these
            colormaps correspond to colors in the color (or self.color
            if you prefer).
        color - 4-list of color strings (default ["tab:red",
            "tab:blue", "tab:green", "tab:orange"]), these are used
            for plotting energy density in this order: dipolar,
            anisotropy, Zeeman, total.
        line_width - float (default 1.5), line width of energy
            densities in plots.
        figs_size - 4-list of 2-tuples of ints or floats (default
            [(4, 3), (4, 3), (6, 4), (6, 4)]), [inch] each tuple
            represents figsize parameter for each plot in this order:
            rectilinear, polar, heff, beff.
        elabels - 4-list of strings, legend labels for energy density
            in the same order as e.g. in color.
        hlabels - 5-list of strings, legend labels for effective
            magnetic field ant its components in the same order as
            e.g. in colors, plus for effective field in energy
            minimum.
        blabels - 4-list of strings, legend labels for effective
            magnetic induction and its components in the same order as
            e.g. in colors.
        angles - 4-list of strings, angle names used in plots in this
            order: delta, xi, phi, phi_0.
        pe - list of path_effects objects, path effects for advanced
            formatting of text plotted using fit_angle and
            plot_other_angles.  By default this creates a 0.5-thick
            white outline around the letters (for better clarity of
            drawn angle values).
        suffixes - 5-list of str, file suffixes to append to loc+name
            in the following order:
            metadata, energy density rectilinear plot, energy density
            polar, effective field plot, effective induction plot.
        ubrac - string, ususally one of {" ({})", " [{}]"}, format of
            unit braces. Must include {} where to place the units.
        polcorr - 4-list of float, [1] correction of position of the
            polar plot radial floating axis as fractions of figure
            width and height as [left, bottom, width, height].
        hxlimshift - 2-list of float, [mT] shift of x limits in the
            heff plot.
        bxlimshift - 2-list of float, [mT] shift of x limits in the
            beff plot.

    Methods:
        compute_energy_density(self, disp_mess_fmin=False)
            - calculates all the necessary energy density components
            and its minimum position.
            Arguments:
                disp_mess_fmin - bool, whether to print minimization
                    procedure output messages into terminal.
        compute_effective_field(self) - calculates all the necessary
            magnetic field components (as mu_0*H, that is, in units of
            induction).
        save_metadata(self) - saves all calculation parameters and
            results into a TXT file.
        plot_rectilinear(self) - generates a plot of energy density
            and its components in rectilinear projection.
        plot_polar(self) - generates a plot of energy density and
            its components in polar projection.
            Note: This plots only positive values of energy!  For
            expected negative values, rectilinear projection is
            recommended.
        plot_heff(self) - generates a quiver plot of effective
            magnetic field mu_0*H_eff and its components.  If
            fit_angle is True, this also plots effective magnetic
            field for angle phi_0 (angle of magnetization from x axis
            at energy density minimum).
        plot_beff(self) - generates a quiver plot of effective
            magnetic induction B_eff = mu_0*(H_eff + M) and its
            components.  If fit_angle is True, also plots effective
            magnetic induction for angle phi_0 (angle of
            magnetization from x axis at energy density minimum).
        reset_computation(self) - resets calculated data in order to
            do some calculation procedures repeatably with changed
            parameters. Affects values of:
            phi, eden, htot, hdip, hani, phi_emin, emin, h_emin.
        evaluate(self) - automatic processiing of the calculation and
            plotting according to the calculation setup.
    """
    def __init__(self, msat, name, loc="", dpi=250, npoints=100,
                 use_dip=False, use_uniax=False, use_bext=False,
                 plot_total=False, fit_angle=True, plot_other_angles=True,
                 plot_rectilinear=True, plot_polar=False,
                 plot_heff=True, plot_beff=True, save_pdf=False,
                 title=None, save_metadata=True,
                 demag_factors=(0., 0., 1.),
                 ku=0., delta=0.,
                 bext=0., xi=0., disp_messages=True):
        self.msat = msat
        self.name = name
        if loc.find("\\") > -1:
            raise Exception("Location parameter includes backslashes! "
                            + "Replace them with slashes.")
        elif loc != "":
            self.loc = loc.rstrip("/") + "/"
        else:
            self.loc = loc
        self.dpi = dpi
        self.n = npoints
        self.use = [use_dip, use_uniax, use_bext]
        self.plot = [plot_total, fit_angle, plot_other_angles,
                     plot_rectilinear, plot_polar, plot_heff, plot_beff,
                     save_pdf]
        self.title = title
        self.metadata = save_metadata
        self.demfs = demag_factors
        self.ku = ku
        self.delta = delta
        self.bext = bext
        self.xi = xi
        self.mess = disp_messages
        # phi vector - [rad] angles of magnetization to compute at
        self.phi = np.linspace(0, 2*np.pi, self.n)
        # preallocation for other used variables
        self.eden = np.zeros((4, self.n))  # = [edip, eani, ezee, etot]
        self.htot = np.zeros((2, self.n))
        self.hdip = np.zeros((2, self.n))
        self.hani = np.zeros((2, self.n))
        self.phi_emin = None  # exact phi at minimal total energy density phi_0
        self.emin = None  # exact energy density value at phi_0
        self.h_emin = None  # exact eff. field value at phi_0 (x and y value)
        # definition of colours
        self.colors = [plt.cm.Reds(np.linspace(0, 1, self.n)),
                       plt.cm.Blues(np.linspace(0, 1, self.n)),
                       plt.cm.Greens(np.linspace(0, 1, self.n)),
                       plt.cm.Oranges(np.linspace(0, 1, self.n))]
        self.color = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
        # plot setup variables
        self.line_width = 1.5
        self.figs_size = [(4, 3), (4, 3), (6, 4), (6, 4)]  # rect/pol/heff/beff
        self.elabels = [r"$\epsilon$",
                        r"$\epsilon_{\mathrm{dip}}$",
                        r"$\epsilon_{\mathrm{ani}}$",
                        r"$\epsilon_{\mathrm{Zee}}$",
                        r"$\epsilon_{\mathrm{tot}}$"]
        self.hlabels = [r"$\mu_0 H_{\mathrm{dip}}$",
                        r"$\mu_0 H_{\mathrm{ani}}$",
                        r"$\mu_0 H_{\mathrm{ext}}$",
                        r"$\mu_0 H_{\mathrm{eff}}$",
                        r"$\mu_0 H_{\mathrm{eff}}$"
                        + r"$|_{\epsilon=\epsilon_{\mathrm{min}}}$",
                        r"$\mu_0 H_{\mathrm{eff}}(\varphi_0)$"]
        self.blabels = [r"$\mu_0 (H_{\mathrm{dip}}+M)$",
                        r"$\mu_0 (H_{\mathrm{ani}}+M)$",
                        r"$\mu_0 (H_{\mathrm{ext}}+M)$",
                        r"$\mu_0 (H_{\mathrm{eff}}+M)$"]
        self.angles = [r"$\delta$", r"$\xi$", r"$\varphi$", r"$\varphi_0$"]
        self.pe = [mpe.Stroke(linewidth=0.5, foreground='w'), mpe.Normal()]
        self.suffixes = ["_metadata", "_eden_rect", "_eden_polar", "_heff",
                         "_beff"]
        self.ubrac = " ({})"
        self.polcorr = [0, 0, 0, 0]
        self.hxlimshift = [0, 0]
        self.bxlimshift = [0, 0]

    def compute_energy_density(self, disp_mess_fmin=False):
        """Computes all energy density components and energy-density
         minimum position.
         """
        if self.use[0]:  # dipolar energy density
            self.eden[0] = (MU0/2*self.msat**2
                            * (self.demfs[0]*np.cos(self.phi)**2
                               + self.demfs[1]*np.sin(self.phi)**2))
            self.eden[3] += self.eden[0]
        if self.use[1]:  # anisotropy energy density
            self.eden[1] = self.ku * np.sin(self.phi - self.delta) ** 2
            self.eden[3] += self.eden[1]
        if self.use[2]:  # Zeeman energy density
            self.eden[2] = -self.msat*self.bext*np.cos(self.phi - self.xi)
            self.eden[3] += self.eden[2]
        if self.plot[1]:
            f_etot = interp1d(self.phi, self.eden[3], "cubic")

            def f_etot_sym(phi):
                """Symmetrical interpolation of etot (for each 2pi rad)."""
                return f_etot(np.mod(phi, np.pi*2))

            start_phi = self.phi[np.argmin(self.eden[3])]
            # self.phi_emin = fmin(f_etot, start_phi)[0]  # without symmetry
            fminout = fmin(f_etot_sym, start_phi, full_output=True,
                           disp=disp_mess_fmin)
            self.phi_emin, self.emin = fminout[0][0], fminout[1]
        printif(self.mess, "Energy density calculated.")

    def compute_effective_field(self):
        """Computes all magnetic field components."""
        if self.use[0]:  # dipolar (demagnetizing) field, x and y components
            self.hdip[0] = -MU0*self.demfs[0]*self.msat*np.cos(self.phi)
            self.hdip[1] = -MU0*self.demfs[1]*self.msat*np.sin(self.phi)
            self.htot += self.hdip
        if self.use[1]:  # anisotropy field, x and y components
            self.hani[0] = (2 * self.ku / self.msat
                            * np.cos(self.phi - self.delta)
                            * np.cos(self.delta))
            self.hani[1] = (2 * self.ku / self.msat
                            * np.cos(self.phi - self.delta)
                            * np.sin(self.delta))
            self.htot += self.hani
        if self.use[2]:  # external field, x and y components
            self.htot[0] = self.htot[0] + self.bext*np.cos(self.xi)
            self.htot[1] = self.htot[1] + self.bext*np.sin(self.xi)
        if self.plot[1]:
            f_htotx = interp1d(self.phi, self.htot[0], "cubic")
            f_htoty = interp1d(self.phi, self.htot[1], "cubic")
            self.h_emin = [f_htotx(self.phi_emin), f_htoty(self.phi_emin)]
        printif(self.mess, "Effective field calculated.")

    def save_metadata(self):
        """Saves all parameters of the calculation."""
        plot_info = ("plot total", "fit angle", "add title",
                     "plot rectilinear", "plot polar", "plot heff",
                     "plot beff", "save pdf")
        use_info = ("dipolar", "uniaxial anisotropy", "Zeeman/external")
        with open(self.loc+self.name+self.suffixes[0]+".txt", "w+") as file:
            file.write("Metadata for same-named plot.\n*** Plot setup: ***\n")
            file.write(f"DPI = {self.dpi}\n")
            for i, j in enumerate(self.plot):
                file.write("{}: {}\n".format(plot_info[i], j))
            file.write("*** Model components: ***\n")
            for i, j in enumerate(self.use):
                file.write("{}: {}\n".format(use_info[i], j))
            file.write("*** Variables: ***\n")
            file.write(f"M_sat [A/m] = {self.msat}\n")
            file.write(f"comp. nodes = {self.n}\n")
            file.write(f"demag. tensor N diagonal [] = {self.demfs}\n")
            file.write(f"Tr(N) [] (1 expected) = {np.sum(self.demfs)}\n")
            file.write(f"K_u [J/m^3] = {self.ku}\n")
            file.write(f"delta [rad] = {self.delta}\n")
            file.write(f"mu_0*H_ext [T] = {self.bext}\n")
            file.write(f"xi [rad] = {self.xi}\n")
            file.write("*** Results of computation: ***\n")
            file.write(f"phi for epsilon_min [rad] = {self.phi_emin}\n")
            if self.plot[1]:
                file.write(f"epsilon_min [kJ/m^3] = {self.emin*1e-3}\n")
                heff = np.sqrt(self.h_emin[0]**2 + self.h_emin[1]**2)
                file.write(f"mu_0*H_eff [mT] = {heff*1e3}\n")
                # H_eff and M should be parallel in energy minimum => sum
                file.write(f"B_eff [T] = {heff+MU0*self.msat}\n")
        printif(self.mess, "Metadata saved.")

    def plot_rectilinear(self):
        """Creates and saves a dependence plot of energy density
        epsilon on magnetization direction angle phi in rectilinear
        projection.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figs_size[0],
                               constrained_layout=True)
        for i, j in enumerate(self.use):
            if j:
                ax.plot(self.phi, self.eden[i], "-", c=self.color[i],
                        lw=self.line_width, label=self.elabels[i+1])
        if self.plot[0]:  # plot total energy density
            ax.plot(self.phi, self.eden[3], "-", c=self.color[3],
                    lw=self.line_width, label=self.elabels[4])
        ax.legend(loc="upper right")
        ylims = ax.get_ylim()
        xticklabels = np.arange(0, 361, 45)
        xticks = xticklabels*np.pi/180
        ax.set_xticks(xticks, xticklabels)
        ax.set_xlabel(self.angles[2]+self.ubrac.format("°"))
        yticks = ax.get_yticks()
        ax.set_yticks(yticks, np.array(yticks)*1e-3)
        ax.set_ylabel(self.elabels[0]+self.ubrac.format(r"kJ/m$^3$"))
        ax.set_ylim(ylims)  # just to make sure limits are not changed
        if self.plot[1]:  # plot angle of energy density minimum
            ax.plot(self.phi_emin * np.ones(2), ylims, "--", c=self.color[3],
                    lw=1)
            ax.text(self.phi_emin, ylims[0] + np.diff(ylims)*0.1,
                    self.angles[3] + r"$=${:.2f}°"
                    .format(self.phi_emin*180/np.pi), path_effects=self.pe)
        if self.plot[2] and self.use[1]:  # plot angle of uni. anisotropy axis
            ax.plot(self.delta * np.ones(2), ylims, "--", c=self.color[1],
                    lw=1)
            ax.text(self.delta, ylims[0] + np.diff(ylims)*0.1
                    * np.sum([1, self.plot[1]]), self.angles[0]
                    + r"$=${:.2f}°".format(self.delta*180/np.pi),
                    path_effects=self.pe)
        if self.plot[2] and self.use[2]:  # plot angle of external field
            ax.plot(self.xi * np.ones(2), ylims, "--", c=self.color[2],
                    lw=1)
            ax.text(self.xi, ylims[0] + np.diff(ylims)*0.1
                    * np.sum([1, self.use[1], self.plot[1]]), self.angles[1]
                    + r"$=${:.2f}°".format(self.xi * 180 / np.pi),
                    path_effects=self.pe)
        if self.title is not None:  # add suptitle to the figure
            fig.suptitle(self.title)
        ax.grid()
        plt.savefig(self.loc+self.name+self.suffixes[1]+".png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+self.suffixes[1]+".pdf")
        plt.close(fig)
        printif(self.mess, "Rectilinear plot saved.")

    def plot_polar(self):
        """Creates and saves a dependence plot of energy density
        epsilon on magnetization direction angle phi in polar
        projection.
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figs_size[1],
                               constrained_layout=True,
                               subplot_kw={"projection": "polar"})
        for i, j in enumerate(self.use):
            if j:
                ax.plot(self.phi, self.eden[i], "-", c=self.color[i],
                        lw=self.line_width, label=self.elabels[i+1])
        if self.plot[0]:  # plot total energy density
            ax.plot(self.phi, self.eden[3], "-", c=self.color[3],
                    lw=self.line_width, label=self.elabels[4])
        ax.legend(loc="lower left", bbox_to_anchor=(1.1, -0.2))
        rlims = ax.get_ylim()
        rheights = (0.8, 0.6, 0.4)
        if self.plot[1]:  # plot angle of energy density minimum
            ax.plot([self.phi_emin, 0], [rlims[1], 0], "--", lw=1,
                    c=self.color[3])
            ax.plot(np.linspace(0, self.phi_emin, 50),
                    np.ones(50)*rlims[1]*rheights[0], "-k", lw=0.5)
            ax.text(self.phi_emin/2, rlims[1]*(rheights[0]+0.02),
                    "{:.1f}°".format(self.phi_emin*180/np.pi),
                    path_effects=self.pe)
        if self.plot[2] and self.use[1]:  # plot angle of uni. anisotropy axis
            ax.plot([self.delta, 0, self.delta+np.pi], [rlims[1], 0, rlims[1]],
                    "--", lw=1, c=self.color[1])
            ax.plot(np.linspace(0, self.delta, 50),
                    np.ones(50) * rlims[1] * rheights[1], "-k", lw=0.5)
            ax.text(self.delta / 2, rlims[1] * (rheights[1] + 0.02),
                    "{:.1f}°".format(self.delta * 180 / np.pi),
                    path_effects=self.pe)
        if self.plot[2] and self.use[2]:  # plot angle of external field
            ax.plot([self.xi, 0, self.xi], [rlims[1], 0, rlims[1]], "--",
                    lw=1, c=self.color[2])
            ax.plot(np.linspace(0, self.xi, 50),
                    np.ones(50) * rlims[1] * rheights[2], "-k", lw=0.5)
            ax.text(self.xi / 2, rlims[1] * (rheights[2] + 0.02),
                    "{:.1f}°".format(self.xi * 180 / np.pi),
                    path_effects=self.pe)
        if self.title is not None:  # add suptitle to the figure
            fig.suptitle(self.title)
        if rlims[0] != 0:
            rlims = (0, rlims[1])
        ax.set_ylim(rlims)
        rticks = ax.get_yticks()
        if rticks[0] != 0:
            rticks = np.zeros(len(rticks) + 1)
            rticks[1:] = ax.get_yticks()
        rticklabels = rticks / 1000  # convert [J/m3]->[kJ/m3]
        ax.set_yticklabels([])
        box = ax.get_position()
        if self.title is not None:
            hei = 0.5
        else:
            hei = 0.55
        axl = fig.add_axes([box.xmax+box.xmin/2+self.polcorr[0],
                            hei*(box.ymin+box.ymax)+self.polcorr[1],
                            box.width/30+self.polcorr[2],
                            box.height*0.5+self.polcorr[3]])
        axl.spines['top'].set_visible(False)
        axl.spines['right'].set_visible(False)
        axl.spines['bottom'].set_visible(False)
        axl.yaxis.set_ticks_position('both')
        axl.xaxis.set_ticks_position('none')
        axl.set_xticklabels([])
        axl.set_yticks(rticks, rticklabels)
        axl.tick_params(right=False)
        axl.set_ylim(rlims)
        axl.set_ylabel(self.elabels[0]+self.ubrac.format(r"kJ/m$^3$"))
        ax.set_xlabel(self.angles[2])
        plt.savefig(self.loc+self.name+self.suffixes[2]+".png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+self.suffixes[2]+".pdf")
        plt.close(fig)
        printif(self.mess, "Polar plot saved.")

    def plot_heff(self):
        """Creates and saves a dependence plot of effective magnetic
        field vector mu0*H_eff on magnetization direction angle phi
        (shown by color, for phi = 0, ..., 2pi color changes light to
        dark).
        """
        cu = 1e3  # convert units (e.g. T -> mT for labelling)
        # should be multiplied by the number of possible elements to fit (4)
        scale = (np.max(np.sqrt(np.sum(self.htot**2, 0)))*(sum(self.use)+1)
                 + (self.hxlimshift[1]-self.hxlimshift[0])/cu)
        fig = plt.figure(figsize=self.figs_size[2], constrained_layout=True)
        xlims = (0+self.hxlimshift[0], scale*cu+self.hxlimshift[0])
        plt.xlim(xlims)
        zer, width, pose = np.zeros(self.n), 0.003, 0.5
        step = (xlims[1]-self.hxlimshift[1])/(sum(self.use)+1)
        if self.use[0]:  # plot vectors of dipolar field
            plt.quiver(zer+step*pose, zer, self.hdip[0], self.hdip[1],
                       color=self.colors[0], scale=scale, width=width,
                       label=self.hlabels[0])
            pose += 1
        if self.use[1]:  # plot vectors of anisotropy field
            plt.quiver(zer+step*pose, zer, self.hani[0], self.hani[1],
                       color=self.colors[1], scale=scale, width=width,
                       label=self.hlabels[1])
            pose += 1
        if self.use[2]:  # plot vector(s) of external field
            plt.quiver(zer+step*pose, zer, zer+self.bext*np.cos(self.xi),
                       zer+self.bext*np.sin(self.xi),
                       color=self.color[2], scale=scale, width=width,
                       label=self.hlabels[2])
            pose += 1
        if self.plot[0]:  # plot vectors of total effective field
            plt.quiver(zer+step*pose, zer, self.htot[0], self.htot[1],
                       color=self.colors[3], scale=scale, width=width,
                       label=self.hlabels[3])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")
        plt.xlabel(r"$\mu_0 H_x$"+self.ubrac.format("mT"))
        plt.ylabel(r"$\mu_0 H_y$"+self.ubrac.format("mT"))
        if self.plot[0] and self.plot[1]:
            plt.quiver(step*pose, 0, self.h_emin[0], self.h_emin[1], color="k",
                       scale=scale, width=width, label=self.hlabels[4])
            plt.text(0.6, 0.9, self.hlabels[4] + r"$= {:.3f}\,$mT"
                     .format(np.sqrt(self.h_emin[0]**2+self.h_emin[1]**2)*1e3),
                     transform=ax.transAxes, path_effects=self.pe)
            plt.text(0.6, 0.8, self.angles[3] + r"$ = {:.3f}$°"
                     .format(np.arctan2(self.h_emin[1], self.h_emin[0])
                             / np.pi*180), transform=ax.transAxes,
                     path_effects=self.pe)
        leg = plt.legend(loc="upper left")
        sup_i = 0
        for i, j in enumerate(leg.legendHandles[:np.sum(self.use)
                                                + self.plot[0]]):
            if i < len(self.use) and not self.use[i]:
                sup_i += 1
            j.set_color(self.color[i+sup_i])
        if self.title is not None:
            plt.title(self.title)
        plt.savefig(self.loc+self.name+self.suffixes[3]+".png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+self.suffixes[3]+".pdf")
        plt.close(fig)
        printif(self.mess, "Effective field plot saved.")

    def plot_beff(self):
        """Creates and saves a dependence plot of effective magnetic
        induction vector B_eff on magnetization direction angle phi
        (shown by color, for phi = 0, ..., 2pi color changes light to
        dark).
        """
        mx, my = MU0*self.msat*np.cos(self.phi), MU0*self.msat*np.sin(self.phi)
        cu = 1e3  # convert units (e.g. T -> mT for labelling)
        # should be multiplied by the number of possible elements to fit (4)
        scale = (np.max(np.sqrt(np.sum((self.htot
                                        + np.vstack((mx, my)))**2, 0)))
                 * (sum(self.use)+1)
                 + (self.bxlimshift[1]-self.bxlimshift[0])/cu)
        fig = plt.figure(figsize=self.figs_size[3], constrained_layout=True)
        xlims = (0+self.bxlimshift[0], scale*cu+self.bxlimshift[0])
        plt.xlim(xlims)
        zer, width, pose = np.zeros(self.n), 0.003, 0.5
        step = (xlims[1]-self.bxlimshift[1])/(sum(self.use)+1)
        if self.use[0]:  # plot vectors of dipolar field
            plt.quiver(zer+step*pose, zer, self.hdip[0]+mx, self.hdip[1]+my,
                       color=self.colors[0], scale=scale, width=width,
                       label=self.blabels[0])
            pose += 1
        if self.use[1]:  # plot vectors of anisotropy field
            plt.quiver(zer+step*pose, zer, self.hani[0]+mx, self.hani[1]+my,
                       color=self.colors[1], scale=scale, width=width,
                       label=self.blabels[1])
            pose += 1
        if self.use[2]:  # plot vector(s) of external field
            plt.quiver(zer+step*pose, zer, zer+self.bext*np.cos(self.xi)+mx,
                       zer+self.bext*np.sin(self.xi)+my,
                       color=self.colors[2], scale=scale, width=width,
                       label=self.blabels[2])
            pose += 1
        if self.plot[0]:  # plot vectors of total effective field
            plt.quiver(zer+step*pose, zer, self.htot[0]+mx, self.htot[1]+my,
                       color=self.colors[3], scale=scale, width=width,
                       label=self.blabels[3])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")
        plt.xlabel(r"$B_x$"+self.ubrac.format("mT"))
        plt.ylabel(r"$B_y$"+self.ubrac.format("mT"))
        f_btotx = interp1d(self.phi, self.htot[0]+mx, "cubic")
        f_btoty = interp1d(self.phi, self.htot[1]+my, "cubic")
        if self.plot[0] and self.plot[1]:
            plt.quiver(step*pose, 0, f_btotx(self.phi_emin),
                       f_btoty(self.phi_emin), color="k",
                       scale=scale, width=width, label=self.hlabels[4])
            plt.text(0.6, 0.9, self.hlabels[4] + r"$= {:.3f}\,$mT"
                     .format(np.sqrt(f_btotx(self.phi_emin)**2
                                     + f_btoty(self.phi_emin)**2)*1e3),
                     transform=ax.transAxes, path_effects=self.pe)
            plt.text(0.6, 0.8, self.angles[3] + r"$ = {:.3f}$°"
                     .format(np.arctan2(f_btoty(self.phi_emin),
                                        f_btotx(self.phi_emin))/np.pi*180),
                     transform=ax.transAxes, path_effects=self.pe)
        leg = plt.legend(loc="upper left")
        sup_i = 0
        for i, j in enumerate(leg.legendHandles[:np.sum(self.use)
                                                + self.plot[0]]):
            if i < len(self.use) and not self.use[i]:
                sup_i += 1
            j.set_color(self.color[i + sup_i])
        if self.title is not None:
            plt.title(self.title)
        plt.savefig(self.loc+self.name+self.suffixes[4]+".png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+self.suffixes[4]+".pdf")
        plt.close(fig)
        printif(self.mess, "Effective inducion plot saved.")

    def reset_computation(self):
        """Method for resetting calculated data in order to do some
        calculation procedures repeatably with changed parameters.
        Affects values of:
        phi, eden, htot, hdip, hani, phi_emin, emin, h_emin.
        """
        # phi vector - [rad] angles of magnetization to compute at
        self.phi = np.linspace(0, 2*np.pi, self.n)
        # preallocation for other used variables
        self.eden = np.zeros((4, self.n))  # = [edip, eani, ezee, etot]
        self.htot = np.zeros((2, self.n))
        self.hdip = np.zeros((2, self.n))
        self.hani = np.zeros((2, self.n))
        self.phi_emin = None  # exact phi at minimal total energy density phi_0
        self.emin = None  # exact energy density value at phi_0
        self.h_emin = None  # exact eff. field value at phi_0 (x and y value)

    def evaluate(self):
        """Method for automatic processing of the calculations and
        plotting.  Calls all methods in the right order.  The same can
        be done manually for more versatility.
        """
        self.compute_energy_density()
        self.compute_effective_field()
        if self.plot[3]:
            self.plot_rectilinear()
        if self.plot[4]:
            self.plot_polar()
        if self.plot[5]:
            self.plot_heff()
        if self.plot[6]:
            self.plot_beff()
        if self.metadata:
            self.save_metadata()


class Hysteresis:
    """Class for calculating Stoner-Wohlfarth-like hysteresis loops
    based on EpsminHeff class.  Due to EpsminHeff, the model is
    restricted to xy plane.

    Keyword Args:
        ehobj - EpsminHeff object, serves as a starting point in
            calculations.  Most parameters (e.g. n, msat, xi, ...) are
            acquired from it.
        bext_max - float or None (default None), [T] maximum applied
            external field.  Determines hysteresis-loop limits as
            <-bext_max; +bext_max>.  If None, determined from
            ehobj.bext.
        loopn - int (default 300), number of calculation points in the
            hysteresis loop.
        plot_hyst_loop - bool (default True), whether to plot the
            final hysteresis loop using plot_loop() method.
        plot_eden_profile - bool (default False), whether to plot total
            energy density profiles for each B_ext (i.e. loopn-times).
            Useful for small loopn (up to ca. 30). For larger values,
            the plot might not be readable.
        eden_cmap - matplotlib.cm object, matplotlib colormap
            object used for energy density profiles.
        disp_messages - bool (default False), whether to print status
            messages into terminal. Applies to all messages except the
            ones from ehobj (it has its own disp_messages kwarg).

    Stored Values:
        (for kwargs info see above)
        ehobj - ehobj init kwarg.
        loopn - loopn init kwarg.
        bext_max - bext_max init kwarg.
        bexts - 1D numpy array, [T] array of B_ext values for the
            hysteresis loop starting at -bext_max (see init kwarg),
            bexts.shape is (loopn, ).
        plot_hyst_loop - plot_hyst_loop init kwarg.
        plot_eden_profile - plot_eden_profile init kwarg.
        edp_cmap - eden_cmap init kwarg.
        mess - disp_messages init kwarg.
        m - 1D numpy array, [1] normalized magnetisation component
            along an axis defined by ehobj.xi, m.shape is (loopn, ).
            Hysteresis-loop data are stored here.
        hl_figsize - 2-tuple of float (default (6.5, 4)), [in]
            hysteresis-loop (HL) figure size.
        hl_kwargs - dict (default {"ls": ".-", "marker": ".", "lw": 1.5,
            "c": "tab:blue", "ms": 1.5}), keyword arguments to pass
            into plt.plot() of the HL plot.
        hl_legend - str or None (default None), if str, legend will be
            displayed and this string defines the legend location of
            the HL plot. If None, legend is not displayed.
        hl_leglabel - str (default r"$\\xi=${}$\\,$°"
            .format(self.ehobj.xi)), legend label for the HL plot.
            Note that it will be displayed only if hl_legend is a valid
            matplotlib legend location, that is, the loc kwarg.
        hl_axlabels - 2-list of str (default
            [r"$\\mathrm{\\mu}_0H$ [mT]", r"$M/M_{\\mathrm{s}}$ []"]),
            the x and y axis labels for the HL plot.
        hl_savesuffix - str (default "_hystloop"), suffix of the save
            name for the HL plot.
        edp_figsize - 2-tuple of float (default (6.5, 4)), [in]
            energy-density-profiles (EDP) figure size.
        edps - 2D numpy array or None, [J/m^3] list of energy density
            profiles, edps.shape is (loopn, ehobj.n). Available only if
            plot_eden_profile is True. Can be added when calling the
            reset_computation() method after setting plot_eden_profile
            to True.
        edp_peaks - list or None, list of found peaks (numpy arrays of
            int) for each row, that is energy profile, in edps.
        edp_bounds - 2D numpy array or None, [rad] array of bounds for
            each energy profile in edps,
            edp_bounds.shape is (loopn, 2).
        edp_phiemins - 1D numpy array of float, [rad] array of phi_emin
            values for each energy profile in edps.
        edp_plot - 4-list of bool (default [True, True, True, True]),
            whether to plot EDP components in the following order:
            energy profiles, found peaks, calculation areas, found
            minima.
        edp_cmaptype - str, one of {"sequential", "categorical"}
            (default "sequential"), type of edp_cmap. Determines the
            choice of colormap colors: "sequential" as
            edp_cmap(i/loopn), "categorical" as edp_cmap(i % 255),
            where i is the element index of bexts.
        edp_legend - str or None (default None), if str, legend will be
            displayed and this string defines the legend location of
            the EDP plot. If None, legend is not displayed.
        edp_leglabels - str (default "{:.1f} mT"), legend label format
            string for the EDP plot with a pair of curly brackets for
            the B_ext value in militeslas with a valid spec_format.
            Note that it will be displayed only if edp_legend is a valid
            matplotlib legend location, that is, the loc kwarg.
        edp_axlabels - 2-list of str (default [self.ehobj.angles[2]
            + " [°]", self.ehobj.elabels[4] + r" [kJ/m$^3$]"]),
            the x and y axis labels for the EDP plot.
        edp_savesuffix - str (default "_EdenMinima"), suffix of the
            save name for the EDP plot.

    Methods:
        compute_loop(self) - computes the hysteresis loop and energy
            density profiles.
        plot_edp(self) - plots the energy density profiles.
        plot_hl(self) - plots the hysteresis loop.
        reset_computation(self) - resets calculated data in order to
            do some calculation procedures repeatably with changed
            parameters. Affects values of:
            ehobj (see EpsminHeff.reset_computation() method), bexts,
            m, hl_leglabel, edps, edp_peaks, edp_bounds, edp_phiemins,
            edp_axlabels.
        evaluate(self) - automatic processing of the calculation and
            plotting according to the calculation setup.
    """
    def __init__(self, ehobj, loopn=300, bext_max=None, plot_hyst_loop=True,
                 plot_eden_profile=False, eden_cmap=None, disp_messages=False):
        self.ehobj = ehobj
        self.ehobj.plot[1] = True  # necessary for emin calculations
        self.loopn = loopn
        if bext_max is None:
            self.bext_max = self.ehobj.bext
        else:
            self.bext_max = bext_max
        self.bexts = np.hstack((np.linspace(-self.bext_max, self.bext_max,
                                            self.loopn//2),
                                np.linspace(self.bext_max, -self.bext_max,
                                            self.loopn - self.loopn//2)))
        self.plot_hyst_loop = plot_hyst_loop
        self.plot_eden_profile = plot_eden_profile
        self.edp_cmap = eden_cmap
        self.mess = disp_messages
        self.m = np.zeros(self.loopn)
        # ### hysteresis loop plot params
        self.hl_figsize = (6.5, 4)
        self.hl_kwargs = {"ls": "-", "marker": ".", "lw": 1.5, "c": "tab:blue",
                          "ms": 1.5}
        self.hl_legend = None
        self.hl_leglabel = r"$\xi=${}$\,$°".format(self.ehobj.xi)
        self.hl_axlabels = [r"$\mathrm{\mu}_0H$"+self.ehobj.ubrac.format("mT"),
                            r"$M/M_{\mathrm{s}}$"+self.ehobj.ubrac.format("")]
        self.hl_savesuffix = "_hystloop"
        # ### energy density profiles plot params
        self.edp_figsize = (6.5, 4)
        # preallocation for energy density profiles, peaks, bounds, and minima
        if self.plot_eden_profile:
            self.edps = np.zeros((self.loopn, self.ehobj.n))
            self.edp_peaks, self.edp_bounds = [], np.zeros((self.loopn, 2))
            self.edp_phiemins = np.zeros(self.loopn)
        else:
            self.edps, self.edp_peaks, self.edp_bounds = None, None, None
            self.edp_phiemins = None
        # plot: energy profiles, found peaks, calculation areas, found minima
        self.edp_plot = [True, True, True, True]
        self.edp_cmaptype = "sequential"
        self.edp_legend = None
        self.edp_leglabels = "{:.1f} mT"
        self.edp_axlabels = [self.ehobj.angles[2]+self.ehobj.ubrac.format("°"),
                             self.ehobj.elabels[4]
                             + self.ehobj.ubrac.format(r"kJ/m$^3$")]
        self.edp_savesuffix = "_EdenMinima"

    def compute_loop(self):
        """Method for computing the hysteresis loop and energy density
        profiles.
        """
        for i in range(self.loopn):
            if self.ehobj.phi_emin is None:  # compute first value of phi_emin
                self.ehobj.compute_energy_density()
            lastphi_emin = self.ehobj.phi_emin
            self.ehobj.reset_computation()
            self.ehobj.bext = self.bexts[i]
            self.ehobj.compute_energy_density()
            f_etot = interp1d(self.ehobj.phi, self.ehobj.eden[3], "cubic")

            def f_etot_sym(phi):
                """Symmetrical interpolation of f_etot (for each 2pi rad)."""
                return f_etot(np.mod(phi, np.pi * 2))

            # I assumed the energy density has only small amount of extrema and
            #   is quite smooth, so to find them, small prominence is used.
            peaks, props = find_peaks(self.ehobj.eden[3], prominence=1e-3)
            if np.shape(peaks) == (0,) or np.shape(peaks) == (1,):
                # in case of empty list of peaks -> include one edge
                # Also do this for the case of one peak (in special cases this
                #   may prove useful and should not affect normal cases)
                elonetot = np.hstack((self.ehobj.eden[3][-2],
                                      self.ehobj.eden[3]))
                #                       self.ehobj.eden[3][1]))  # second edge
                peaks, props = find_peaks(elonetot, prominence=1e-3)
                peaks = peaks - 1  # get indices of non-elongated e_tot
            printif(self.mess, f"Bext={self.bexts[i]:.05f} T, ",
                    f"peaks: {peaks}, prominences: {props['prominences']}")

            # get peaks as bounds (phi in (a; b))
            # PHI Index Nearest to lastphi_emin
            phiin = np.argmin(np.abs(self.ehobj.phi-lastphi_emin))
            a, b = 0, 2*np.pi  # preallocation for a and b
            midpeakphi = np.linspace(a, b, self.ehobj.n)
            phi_init = midpeakphi[np.argmin(f_etot_sym(midpeakphi))]
            if phiin in peaks and len(peaks) <= 2:
                a, b = 0, 2*np.pi
            elif len(peaks) == 1:  # one peak => one minimum
                if phi_init < self.ehobj.phi[peaks[0]]:
                    a, b = 0, self.ehobj.phi[peaks[0]]
                else:
                    a, b = self.ehobj.phi[peaks[0]], 2*np.pi
            elif phiin in peaks:
                phiinp = np.argwhere(peaks == phiin)  # get phiin index in peaks
                a = self.ehobj.phi[peaks[phiinp - 1]]
                b = self.ehobj.phi[peaks[(phiinp + 1) % len(peaks)]]
            else:  # peaks have length at least 2 and phiin is not in peaks
                if peaks[-1] - self.ehobj.n < phiin < peaks[0]:
                    # phiin between last and first peak (phin around zero)
                    a = self.ehobj.phi[peaks[-1]] - 2 * np.pi
                    b = self.ehobj.phi[peaks[0]]
                else:  # phiin between two of the peaks
                    for j in range(len(peaks) - 1):
                        if peaks[j] < phiin < peaks[j + 1]:
                            a = self.ehobj.phi[peaks[j]]
                            b = self.ehobj.phi[peaks[j + 1]]
            fminout = fminbound(f_etot_sym, a, b, full_output=True,
                                disp=self.mess)
            self.ehobj.phi_emin, self.ehobj.emin = fminout[0], fminout[1]
            printif(self.mess, "phi0", phi_init, "/ phi_emin",
                    self.ehobj.phi_emin, "/ bounds:", a, b)
            self.m[i] = np.cos(self.ehobj.phi_emin - self.ehobj.xi)
            if self.plot_eden_profile:
                self.edps[i] = self.ehobj.eden[3]
                self.edp_peaks.append(peaks)
                self.edp_bounds[i] = np.array((a, b))
                self.edp_phiemins[i] = self.ehobj.phi_emin
        printif(self.mess, "Hysteresis loop calculated.")

    def plot_edp(self):
        """Method for plotting the energy density profiles."""

        def cmapi(ii, cmtype=self.edp_cmaptype):
            """Function for identifying the color from colormap."""
            if self.edp_cmap is None:  # in case no colormap is provided
                return "k"
            elif cmtype == "sequential":
                return self.edp_cmap(ii/self.loopn)
            elif cmtype == "categorical":
                return self.edp_cmap(ii % 255)
            else:
                raise Exception("Invalid edp_cmaptype.")

        fig = plt.figure(figsize=self.edp_figsize)
        for i in range(self.loopn):
            if self.edp_plot[0]:  # plot energy profile
                plt.plot(self.ehobj.phi*180/np.pi, self.edps[i]*1e-3, "-",
                         lw=1, c=cmapi(i),
                         label=self.edp_leglabels.format(self.bexts[i]*1e3))
            if self.edp_plot[1]:  # plot found peaks
                plt.plot(self.ehobj.phi[self.edp_peaks[i]]*180/np.pi,
                         self.edps[i][self.edp_peaks[i]]*1e-3, "x", c=cmapi(i))
            if self.edp_plot[2]:  # plot calculation area
                f_etot = interp1d(self.ehobj.phi, self.edps[i], "cubic")

                def etot_sym(phi):
                    """Symmetrical interpolation of etot (for each 2pi rad)."""
                    return f_etot(np.mod(phi, np.pi*2))

                boundphi = np.linspace(self.edp_bounds[i][0],
                                       self.edp_bounds[i][1], self.ehobj.n)
                plt.plot(boundphi*180/np.pi, etot_sym(boundphi)*1e-3,
                         "--", c=cmapi(i), lw=2)
            if self.edp_plot[3]:  # plot energy-density minima
                plt.plot(self.edp_phiemins[i]*180/np.pi,
                         etot_sym(self.edp_phiemins[i])*1e-3, "o", c=cmapi(i),
                         ms=(self.loopn-i)/4+3)
        if self.edp_legend is not None:
            plt.legend(loc=self.edp_legend, fontsize=4)
        plt.xlabel(self.edp_axlabels[0])
        plt.ylabel(self.edp_axlabels[1])
        plt.savefig(self.ehobj.loc+self.ehobj.name+self.edp_savesuffix+".png",
                    dpi=self.ehobj.dpi)
        if self.ehobj.plot[7]:
            plt.savefig(self.ehobj.loc+self.ehobj.name+self.edp_savesuffix
                        + ".pdf")
        plt.close(fig)
        printif(self.mess, "Energy-density-profile plot saved.")

    def plot_hl(self):
        """Method for plotting the hysteresis loop."""
        fig = plt.figure(figsize=self.hl_figsize)
        plt.plot(self.bexts*1e3, self.m, label=self.hl_leglabel,
                 **self.hl_kwargs)
        if self.hl_legend is not None:
            plt.legend(loc=self.hl_legend)
        plt.xlabel(self.hl_axlabels[0])
        plt.ylabel(self.hl_axlabels[1])
        plt.savefig(self.ehobj.loc+self.ehobj.name+self.hl_savesuffix+".png",
                    dpi=self.ehobj.dpi)
        if self.ehobj.plot[7]:
            plt.savefig(self.ehobj.loc+self.ehobj.name+self.hl_savesuffix
                        + ".pdf")
        plt.close(fig)
        printif(self.mess, "Hysteresis-loop plot saved.")

    def reset_computation(self):
        """Method for resetting calculated data in order to do some
        calculation procedures repeatably with changed parameters.
        Affects values of:
        ehobj (see EpsminHeff.reset_computation() method), bexts, m,
        hl_leglabel, hl_axabels, edps, edp_peaks, edp_bounds,
        edp_phiemins, edp_axlabels.
        """
        self.ehobj.reset_computation()
        self.bexts = np.hstack((np.linspace(-self.bext_max, self.bext_max,
                                            self.loopn // 2),
                                np.linspace(self.bext_max, -self.bext_max,
                                            self.loopn - self.loopn // 2)))
        self.m = np.zeros(self.loopn)
        self.hl_leglabel = r"$\xi=${}$\,$°".format(self.ehobj.xi)
        self.hl_axlabels = [r"$\mathrm{\mu}_0H$"+self.ehobj.ubrac.format("mT"),
                            r"$M/M_{\mathrm{s}}$" + self.ehobj.ubrac.format("")]
        if self.plot_eden_profile:
            self.edps = np.zeros((self.loopn, self.ehobj.n))
            self.edp_peaks, self.edp_bounds = [], np.zeros((self.loopn, 2))
            self.edp_phiemins = np.zeros(self.loopn)
        else:
            self.edps, self.edp_peaks, self.edp_bounds = None, None, None
            self.edp_phiemins = None
        self.edp_axlabels = [self.ehobj.angles[2]+self.ehobj.ubrac.format("°"),
                             self.ehobj.elabels[4]
                             + self.ehobj.ubrac.format(r"kJ/m$^3$")]

    def evaluate(self):
        """Method for automatic processing of the calculations and
        plotting.  Calls all methods in the right order.  The same can
        be done manually for more versatility.
        """
        self.compute_loop()
        if self.plot_eden_profile:
            self.plot_edp()
        if self.plot_hyst_loop:
            self.plot_hl()


def printif(condition, *messages, **kwargs):
    """Function for printing messages into terminal upon condition.
    It should support the same functionalities as standard print().
    condition - bool, whether to print the message.
    messages - str or (starred) list of str, text(s) to print.
    kwargs - dict, keyword arguments for print() function.
    """
    if condition:
        print(*messages, **kwargs)


def ytilt(theta):
    """Function that returns absolute angle [°] from y axis, e.g.
    ytilt(60*np.pi/180) == 30 == ytilt(120*np.pi/180).
    theta - float, [rad] positive angle from x axis.
    """
    return np.abs(theta*180/np.pi-90)


def ku2bani(ku, msat):
    """Function that recalculates uniaxial anisotropy constant to
    equivalent anisotropic magnetic field.
    ku - [J/m3] uniaxial anisotropy constant K_u.
    msat - [A/m] saturation magnetisation M_s.
    Returns:
    bani - [T] anisotropic magnetic field mu_0*H_ani.
    """
    return 2*ku/msat


def bani2ku(bani, msat):
    """Function that recalculates anisotropic magnetic field to
    equivalent uniaxial anisotropy constant.
    bani - [T] anisotropic magnetic field mu_0*H_ani.
    msat - [A/m] saturation magnetisation M_s.
    Returns:
    ku - [J/m3] uniaxial anisotropy constant K_u.
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


def wysin_cylinder(h, r):
    """Returns diagonal components Nx, Ny, Nz of the demagnetization
    tensor from Wysin's solution for a homogenously magnetized
    cylinder with rotational axis along z drection.  For more info
    see: https://www.phys.ksu.edu/personal/wysin/notes/demag.pdf
    h - float or ndarray, [m] height of the cylinder.
    r - float or ndarray, [m] radius of the
        cylinder's circular base. If h and r are ndarrays, they
        must have the same size.
    """
    nxy = 1/2/h*(np.sqrt(h**2 + r**2) - r)
    nz = 1/h*(h + r - np.sqrt(h**2 + r**2))
    return nxy, nxy, nz


if __name__ == "__main__":
    main()
