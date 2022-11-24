import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import myfxs as mf


def main():
    """Script for drawing energy density angular dependence, based on uniaxial
    anisotropy constants and/or Aharoni model of rectangular prism. Note that
    it accounts for Bext=0."""
    # script setup
    useaharoni, useuniax, plottotal, fitangle = True, True, True, True
    usebext = False  # not yet included in B_eff
    addtitle, savepdf, plotbeff = True, False, True
    name, dpi = "ani202211118_00_square_test10x10", 250
    # model setup
    dims = (10e-6, 10e-6, 10e-9)  # [m] dimensions of the WG in xyz order
    msat = 830e3  # [A/m] saturation magnetization of the magnetic WG/film
    ku = bani2ku(20e-3, msat)  # [J/m3] uniaxial anisotropy constant
    # [rad] tilt of uniaxial anisotropy from x axis
    tiltuni = (90-20)*np.pi/180
    bext = 5e-3  # [T] external magnetic induction B_ext
    ksi = 90*np.pi/180  # [rad] tilt of B_ext from x axis

    mu0 = 4*np.pi*1e-7  # [H/m] permeability of vacuum
    phi = np.linspace(0, 2*np.pi, 100)  # coordinates for plot, direction of M
    etot = np.zeros(phi.shape)

    mf.makemyfontnice()  # for serif font in plots
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), constrained_layout=True,
                           subplot_kw={"projection": "polar"})
    # choose rectilinear or polar axis projection. Do this better!
    nx, ny, nz, tilttot = 0, 0, 0, 0  # precaution for undefined variables
    if useaharoni:
        nx = aharoni(dims[1], dims[2], dims[0])
        ny = aharoni(dims[2], dims[0], dims[1])
        nz = aharoni(*dims)
        print("Aharoni demag factors:" +
              f"\nNx = {nx}\nNy = {ny}\nNz = {nz}\ncheck sum = {nx+ny+nz}" +
              " (should be equal to 1)")
        edip = mu0/2*msat**2*(nx*np.cos(phi)**2+ny*np.sin(phi)**2)
        ax.plot(phi, edip, "-", c="tab:red", lw=1.5,
                label=r"$\epsilon_{\mathrm{dip}}$")
        etot += edip
    if useuniax:
        eani = ku*np.sin(phi-tiltuni)**2
        ax.plot(phi, eani, "-", c="tab:blue", lw=1.5,
                label=r"$\epsilon_{\mathrm{ani}}$")
        etot += eani
    if usebext:
        ezee = - msat*bext*np.cos(phi-ksi)
        ax.plot(phi, ezee, "-", c="tab:green", lw=1.5,
                label=r"$\epsilon_{\mathrm{Z}}$")
        etot += ezee
    if plottotal:
        ax.plot(phi, etot, "-", c="tab:orange", lw=1.5,
                label=r"$\epsilon_{\mathrm{tot}}$")
    ax.legend(loc="lower left", bbox_to_anchor=(1.1, 0.))
    rlims = ax.get_ylim()
    if fitangle and useuniax and useaharoni:
        ax.plot([tiltuni, 0, tiltuni+np.pi], [rlims[1], 0, rlims[1]], "--",
                lw=1, c="tab:blue")
        rheight0, rheight1 = 0.6, 0.8
        ax.plot(np.linspace(tiltuni, np.pi/2, 50),
                np.ones(50)*rlims[1] * rheight0,
                "-k", lw=1)
        ax.text(np.pi / 2, rlims[1] * (rheight0+0.02),
                "{:.1f}°".format(ytilt(tiltuni)))
        popt, pcov = curve_fit(anicurve, phi, etot,
                               p0=(rlims[1]/2, rlims[1]/2, 1),
                               bounds=((0, 0, 0),
                                       (rlims[1], rlims[1], 2*np.pi)))
        perr = np.sqrt(np.diag(pcov))
        for i in range(len(popt)):
            perr[i], ind = mf.zaoknem(perr[i])
            popt[i] = np.round(popt[i], ind)
        print("fitting parameters and uncertainty:", popt, perr, sep="\n")
        r2 = mf.getr2(etot, anicurve(phi, *popt))
        print("fitted with R2 = {:.04%}".format(r2))
        tilttot = np.pi - popt[2]
        ax.plot([tilttot, 0, tilttot+np.pi], [rlims[1], 0, rlims[1]], "--",
                lw=1, c="tab:orange")
        ax.plot(np.linspace(tilttot, np.pi / 2, 50),
                np.ones(50)*rlims[1]*rheight1, "-k", lw=1)
        ax.text(np.pi / 2, rlims[1] * (rheight1+0.02),
                "{:.1f}°".format(ytilt(tilttot)))
    if addtitle:
        fig.suptitle(f"Calculated for {dims[0]*1e6}×{dims[1]*1e6}" +
                     f"×{dims[2]*1e6}" +
                     r"$\,\mathrm{\mu m}^3$ rect. prism", wrap=False,
                     size="medium")
    # ### test block start (adding "floating" axis next to the plot)
    if rlims[0] != 0:
        rlims = (0, rlims[1])
    ax.set_ylim(rlims)
    rticks = ax.get_yticks()
    if rticks[0] != 0:
        rticks = np.zeros(len(rticks)+1)
        rticks[1:] = ax.get_yticks()
    rticklabels = rticks/1000  # convert [J/m3]->[kJ/m3]
    print(rticks, rticklabels)
    ax.set_yticklabels([])
    # ax.spines["inner"].set_position(("axes", 0))
    box = ax.get_position()
    axl = fig.add_axes([box.xmax+box.xmin/2, 0.5*(box.ymin+box.ymax),
                        box.width/30, box.height*0.5])
    axl.spines['top'].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.spines['bottom'].set_visible(False)
    axl.yaxis.set_ticks_position('both')
    axl.xaxis.set_ticks_position('none')
    axl.set_xticklabels([])
    axl.set_yticks(rticks)
    axl.set_yticklabels(rticklabels)
    axl.tick_params(right=False)
    axl.set_ylim(rlims)
    axl.set_ylabel(r"$\epsilon$ [kJ/m$^3$]")
    # ### test block end
    plt.savefig(f"aniPolar/{name}.png", dpi=dpi)
    if savepdf:
        plt.savefig(f"aniPolar/{name}.pdf")
    plt.close(fig)
    print("Figure saved.")
    if plotbeff and fitangle and useuniax and useaharoni:
        # another testing block for calculating B_eff
        col1 = plt.cm.Blues(np.linspace(0, 1, len(phi)))
        col2 = plt.cm.Greens(np.linspace(0, 1, len(phi)))
        col3 = plt.cm.Reds(np.linspace(0, 1, len(phi)))
        bdx, bdy = -mu0*nx*msat*np.cos(phi), -mu0*ny*msat*np.sin(phi)  # B_dem
        bax = 2*ku/msat*np.cos(phi-tiltuni)*np.cos(tiltuni)  # B_ani
        bay = 2*ku/msat*np.cos(phi-tiltuni)*np.sin(tiltuni)
        bex, bey = bdx + bax, bdy + bay  # B_eff
        equi = np.argmin(np.abs(phi-tilttot))  # index at energy equilibrium
        zer, step, width = np.zeros(phi.shape), 0.5, 0.003
        scale = np.max(np.sqrt(bey**2+bex**2))*3
        fig = plt.figure(figsize=(6, 6), constrained_layout=True)
        plt.quiver(zer-step, zer, bdx, bdy, color=col1, scale=scale,
                   label=r"$\vec{B}_{\mathrm{dem}}$", width=width)
        plt.quiver(zer+step, zer, bax, bay, color=col2, scale=scale,
                   label=r"$\vec{B}_{\mathrm{ani}}$", width=width)
        quiv = plt.quiver(zer, zer, bex, bey, color=col3, scale=scale,
                          label=r"$\vec{B}_{\mathrm{eff}}$", width=width)
        plt.quiver(0, 0, bex[equi], bey[equi], color="k", width=width,
                   label=r"$\vec{B}_{\mathrm{eff}}$ at equilibrium",
                   scale=scale)
        plt.text(bex[equi]/scale*2, bey[equi]/scale*2,
                 r"$B = ${:.3f} mT".format(1e3*np.sqrt(bex[equi]**2 +
                                                       bey[equi]**2)))
        plt.text(bex[equi]/scale*2, bey[equi]/scale*1.8,
                 r"$\theta = ${:.3f}°"
                 .format((np.pi/2-np.arctan(bey[equi]/bex[equi]))/np.pi*180))
        plt.quiverkey(quiv, -step, -step, 10e-3,
                      r"$|\vec{B}| = 10\,\mathrm{mT}$",
                      coordinates="data", color="k")
        plt.legend(loc="upper left")
        plt.xlim((-step*2, step*2))
        plt.ylim((-step*2, step*2))
        plt.title(r"$\vec{B}_{\mathrm{eff}}$ for $\vec{M}$ pointing from 0 " +
                  r"rad to $2\pi$ rad (light to dark) and its parts",
                  size="medium")
        plt.savefig(f"aniPolar/{name}_Beff.png", dpi=dpi)
        plt.close(fig)
        print("B_eff plot saved.")
    with open(f"aniPolar/{name}_z.txt", "w+") as file:
        file.write("Metadata for same-named plot.\n")
        file.write(f"useaharoni, useuniax, plottotal, fitangle = {useaharoni}" +
                   f", {useuniax}, {plottotal}, {fitangle}\n")
        if useaharoni:
            file.write(f"prism dimensions [m]: {dims}\n")
            file.write(f"Demagnetization factors:\n{nx}\n{ny}\n{nz}\n")
        if useuniax:
            file.write(f"Msat [A/m]: {msat}\n")
            file.write(f"Ku [J/m3]/[mT]: {ku}/{ku2bani(ku, msat)*1000}\n")
            file.write(f"tilt from x axis [°]: {tiltuni*180/np.pi}\n")
        if fitangle and useuniax and useaharoni:
            file.write(f"final tilt from x axis [°]: {tilttot*180/np.pi}\n")
        if plotbeff and fitangle and useuniax and useaharoni:
            file.write(f"Beff [mT]: {1e3*np.sqrt(bex[equi]**2+bey[equi]**2)}\n")
    print("Metadata saved.")


def aharoni(a, b, c):
    """Returns demagnetization factor Nz. To get Nx and Ny use (twice) the
    cyclic permutation c->a->b->c.
    a, b, c - float, prism overall dimensions in meters"""
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


def ku2bani(ku, msat):
    """Function that recalculates uniaxial anisotropy constant to equivalent
    anisotropic magnetic field.
    ku - anisotropy constant [J/m3]
    msat - saturation magnetisation [A/m]
    return:
    bani - anisotropic magnetic induction [T]
    """
    return 2*ku/msat


def bani2ku(bani, msat):
    """Function that recalculates anisotropic magnetic field to equivalent
    uniaxial anisotropy constant.
    bani - anisotropic magnetic induction [T]
    msat - saturation magnetisation [A/m]
    returns:
    ku - anisotropy constant [J/m3]
    """
    return msat*bani/2


def anicurve(x, a, b, gamma):
    """Function for calculating uniaxial anisotropy of in-plane magnetized
    structures.
    x - ndarray, list of angles
    a - float, amplitude of anisotropy
    b - float, mean value of anisotropy
    gamma - float, phase (also characterizes tilt)
    """
    return b + a*np.sin(x + gamma)**2


def ytilt(theta):
    """Function that returns absolute angle [°] from y axis, e.g.
    ytilt(60*np.pi/180) == 30 == ytilt(120*np.pi/180).
    theta - float, [rad] positive angle from x axis"""
    return np.abs(theta*180/np.pi-90)


if __name__ == "__main__":
    main()
