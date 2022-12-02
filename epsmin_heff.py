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
    aharoni - calculates demagnetization factors of a homogenously
        magnetized rectangular prism using the Aharoni model.
    wysin_cylinder - calculates demagnetization factors of
        a homogenously magnetized cylinder.

@author: Jan Klíma, jan.klima4@vutbr.cz
"""

import numpy as np  # for vectorization
import matplotlib.pyplot as plt  # for plotting
import matplotlib.patheffects as mpe  # for text formatting in plots
from scipy.optimize import fmin  # for energy minima finding
from scipy.interpolate import interp1d  # for sth similar
import myfxs as mf  # see https://github.com/GiovanniKl/MyFxsPackage

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
        serif - bool (default True), whether to plot text in serif
            font family.
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

    Methods:
        compute_energy_density(self) - calculates all the necessary
            energy density components and its minimum position.
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
        evaluate(self) - automatic processiing of the calculation and
            plotting according to the calculation setup.
    """
    def __init__(self, msat, name, loc="", dpi=250, npoints=100,
                 use_dip=False, use_uniax=False, use_bext=False,
                 plot_total=False, fit_angle=True, plot_other_angles=True,
                 plot_rectilinear=True, plot_polar=False,
                 plot_heff=True, plot_beff=True, save_pdf=False, serif=True,
                 title=None, save_metadata=True,
                 demag_factors=(0., 0., 1.),
                 ku=0., delta=0.,
                 bext=0., xi=0.):
        self.msat = msat
        self.name = name
        if loc != "":
            self.loc = loc.rstrip("/") + "/"
        else:
            self.loc = loc
        self.dpi = dpi
        self.n = npoints
        self.use = [use_dip, use_uniax, use_bext]
        self.plot = [plot_total, fit_angle, plot_other_angles,
                     plot_rectilinear, plot_polar, plot_heff, plot_beff,
                     save_pdf]
        if serif:
            mf.makemyfontnice()  # for serif fonts
        self.title = title
        self.metadata = save_metadata
        self.demfs = demag_factors
        self.ku = ku
        self.delta = delta
        self.bext = bext
        self.xi = xi
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
        self.elabels = [r"$\epsilon_{\mathrm{dip}}$",
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

    def compute_energy_density(self):
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
            fminout = fmin(f_etot_sym, start_phi, full_output=True)
            self.phi_emin, self.emin = fminout[0][0], fminout[1]
        print("Energy density calculated.")

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
        print("Effective field calculated.")

    def save_metadata(self):
        """Saves all parameters of the calculation."""
        plot_info = ("plot total", "fit angle", "add title",
                     "plot rectilinear", "plot polar", "plot heff",
                     "plot beff", "save pdf")
        use_info = ("dipolar", "uniaxial anisotropy", "Zeeman/external")
        with open(self.loc+self.name+"_metadata.txt", "w+") as file:
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
            file.write(f"Tr(N) [] (1 expected) = {np.sum(self.demfs)}")
            file.write(f"K_u [J/m^3] = {self.ku}\n")
            file.write(f"delta [rad] = {self.delta}\n")
            file.write(f"mu_0*H_ext [T] = {self.bext}\n")
            file.write(f"xi [rad] = {self.xi}\n")
            file.write("*** Results of computation: ***\n")
            file.write(f"phi for epsilon_min [rad] = {self.phi_emin}\n")
            if self.plot[1]:
                file.write(f"epsilon_min [kJ/m^3] = {self.emin*1e3}\n")
                heff = np.sqrt(self.h_emin[0]**2 + self.h_emin[1]**2)
                file.write(f"mu_0*H_eff [mT] = {heff}\n")
                # H_eff and M should be parallel in energy minimum => sum
                file.write(f"B_eff [T] = {heff+MU0*self.msat}\n")
        print("Metadata saved.")

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
                        lw=self.line_width, label=self.elabels[i])
        if self.plot[0]:  # plot total energy density
            ax.plot(self.phi, self.eden[3], "-", c=self.color[3],
                    lw=self.line_width, label=self.elabels[3])
        ax.legend(loc="upper right")
        ylims = ax.get_ylim()
        xticklabels = np.arange(0, 361, 45)
        xticks = xticklabels*np.pi/180
        ax.set_xticks(xticks, xticklabels)
        ax.set_xlabel(self.angles[2]+" [°]")
        yticks = ax.get_yticks()
        ax.set_yticks(yticks, np.array(yticks)*1e-3)
        ax.set_ylabel(r"$\epsilon$ [kJ/m$^3$]")
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
        plt.savefig(self.loc+self.name+"_eden_rect.png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc + self.name + "_eden_rect.pdf")
        plt.close(fig)
        print("Rectilinear plot saved.")

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
                        lw=self.line_width, label=self.elabels[i])
        if self.plot[0]:  # plot total energy density
            ax.plot(self.phi, self.eden[3], "-", c=self.color[3],
                    lw=self.line_width, label=self.elabels[3])
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
            ax.plot([self.delta, 0, self.delta], [rlims[1], 0, rlims[1]], "--",
                    lw=1, c=self.color[1])
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
        axl = fig.add_axes([box.xmax+box.xmin/2, hei*(box.ymin+box.ymax),
                            box.width/30, box.height*0.5])
        axl.spines['top'].set_visible(False)
        axl.spines['right'].set_visible(False)
        axl.spines['bottom'].set_visible(False)
        axl.yaxis.set_ticks_position('both')
        axl.xaxis.set_ticks_position('none')
        axl.set_xticklabels([])
        axl.set_yticks(rticks, rticklabels)
        axl.tick_params(right=False)
        axl.set_ylim(rlims)
        axl.set_ylabel(r"$\epsilon$ [kJ/m$^3$]")
        ax.set_xlabel(self.angles[2])
        plt.savefig(self.loc+self.name+"_eden_polar.png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc + self.name + "_eden_polar.pdf")
        plt.close(fig)
        print("Polar plot saved.")

    def plot_heff(self):
        """Creates and saves a dependence plot of effective magnetic
        field vector mu0*H_eff on magnetization direction angle phi
        (shown by color, for phi = 0, ..., 2pi color changes light to
        dark).
        """
        cu = 1e3  # convert units (e.g. T -> mT for labelling)
        # should be multiplied by the number of possible elements to fit (4)
        scale = np.max(np.sqrt(np.sum(self.htot**2, 0)))*4
        fig = plt.figure(figsize=self.figs_size[2], constrained_layout=True)
        xlims = (0, scale*cu)
        plt.xlim(xlims)
        zer, step, width = np.zeros(self.n), xlims[1]/(len(self.use)+1), 0.003
        if self.use[0]:  # plot vectors of dipolar field
            plt.quiver(zer+step*0.5, zer, self.hdip[0], self.hdip[1],
                       color=self.colors[0], scale=scale, width=width,
                       label=self.hlabels[0])
        if self.use[1]:  # plot vectors of anisotropy field
            plt.quiver(zer+step*1.5, zer, self.hani[0], self.hani[1],
                       color=self.colors[1], scale=scale, width=width,
                       label=self.hlabels[1])
        if self.use[2]:  # plot vector(s) of external field
            plt.quiver(zer + step * 2.5, zer, zer + self.bext * np.cos(self.xi),
                       zer + self.bext * np.sin(self.xi),
                       color=self.color[2], scale=scale, width=width,
                       label=self.hlabels[2])
        if self.plot[0]:  # plot vectors of total effective field
            plt.quiver(zer+step*3.5, zer, self.htot[0], self.htot[1],
                       color=self.colors[3], scale=scale, width=width,
                       label=self.hlabels[3])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")
        plt.xlabel(r"$\mu_0 H_x$ [mT]")
        plt.ylabel(r"$\mu_0 H_y$ [mT]")
        if self.plot[0] and self.plot[1]:
            plt.quiver(step*3.5, 0, self.h_emin[0], self.h_emin[1], color="k",
                       scale=scale, width=width, label=self.hlabels[4])
            plt.text(0.6, 0.9, self.hlabels[4] + r"$= {:.3f}\,$mT"
                     .format(np.sqrt(self.h_emin[0]**2+self.h_emin[1]**2)*1e3),
                     transform=ax.transAxes, path_effects=self.pe)
            plt.text(0.6, 0.8, self.angles[3] + r"$ = {:.3f}$°"
                     .format(np.arctan2(self.h_emin[1], self.h_emin[0])
                             / np.pi*180), transform=ax.transAxes,
                     path_effects=self.pe)
        leg = plt.legend(loc="upper left")
        # leg = ax.get_legend()
        sup_i = 0
        for i, j in enumerate(leg.legendHandles[:np.sum(self.use)
                                                + self.plot[0]]):
            if i < len(self.use) and not self.use[i]:
                sup_i += 1
            j.set_color(self.color[i+sup_i])
        if self.title is not None:
            plt.title(self.title)
        plt.savefig(self.loc+self.name+"_heff.png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+"_heff.pdf")
        plt.close(fig)
        print("Effective field plot saved.")

    def plot_beff(self):
        """Creates and saves a dependence plot of effective magnetic
        induction vector B_eff on magnetization direction angle phi
        (shown by color, for phi = 0, ..., 2pi color changes light to
        dark).
        """
        mx, my = MU0*self.msat*np.cos(self.phi), MU0*self.msat*np.sin(self.phi)
        cu = 1e3  # convert units (e.g. T -> mT for labelling)
        # should be multiplied by the number of possible elements to fit (4)
        scale = np.max(np.sqrt(np.sum((self.htot
                                       + np.vstack((mx, my)))**2, 0)))*4
        fig = plt.figure(figsize=self.figs_size[3], constrained_layout=True)
        xlims = (0, scale*cu)
        plt.xlim(xlims)
        zer, step, width = np.zeros(self.n), xlims[1]/(len(self.use)+1), 0.003
        if self.use[0]:  # plot vectors of dipolar field
            plt.quiver(zer+step*0.5, zer, self.hdip[0]+mx, self.hdip[1]+my,
                       color=self.colors[0], scale=scale, width=width,
                       label=self.blabels[0])
        if self.use[1]:  # plot vectors of anisotropy field
            plt.quiver(zer+step*1.5, zer, self.hani[0]+mx, self.hani[1]+my,
                       color=self.colors[1], scale=scale, width=width,
                       label=self.blabels[1])
        if self.use[2]:  # plot vector(s) of external field
            plt.quiver(zer + step * 2.5, zer,
                       zer + self.bext * np.cos(self.xi) + mx,
                       zer + self.bext * np.sin(self.xi) + my,
                       color=self.colors[2], scale=scale, width=width,
                       label=self.blabels[2])
        if self.plot[0]:  # plot vectors of total effective field
            plt.quiver(zer+step*3.5, zer, self.htot[0]+mx, self.htot[1]+my,
                       color=self.colors[3], scale=scale, width=width,
                       label=self.blabels[3])
        ax = plt.gca()
        ax.set_aspect("equal", adjustable="datalim")
        plt.xlabel(r"$B_x$ [mT]")
        plt.ylabel(r"$B_y$ [mT]")
        f_btotx = interp1d(self.phi, self.htot[0]+mx, "cubic")
        f_btoty = interp1d(self.phi, self.htot[1]+my, "cubic")
        if self.plot[0] and self.plot[1]:
            plt.quiver(step*3.5, 0, f_btotx(self.phi_emin),
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
        plt.savefig(self.loc+self.name+"_beff.png", dpi=self.dpi)
        if self.plot[7]:
            plt.savefig(self.loc+self.name+"_beff.pdf")
        plt.close(fig)
        print("Effective inducion plot saved.")

    def evaluate(self):
        """Method for automatic processing of the calculations and
        plotting. Calls all methods in the right order.  The same can
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
    r - float or ndarray of the same shape as h, [m] radius of the
        cylinder's circular base.
    """
    nxy = 1/2/h*(np.sqrt(h**2 + r**2) - r)
    nz = 1/h*(h + r - np.sqrt(h**2 + r**2))
    return nxy, nxy, nz


if __name__ == "__main__":
    main()
