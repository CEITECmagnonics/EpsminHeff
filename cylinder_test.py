import numpy as np
import epsmin_heff as eh


def main():
    """Function setting all the parameters and controlling the
    computation.  Example of calling the EpsminHeff class from
    outside. This calculates energy density and effective field in
    a thin cylinder.
    """
    name, dpi = "ani20221125_02_cylinder_test", 250
    loc = "test_plots"
    n = 100
    # thin cylinder -> demag factors from Wysin
    dims = (10e-9, 8e-6)  # [m] dimensions of the cylinder (height, radius)
    demfs = eh.wysin_cylinder(*dims)
    msat = 830e3  # [A/m] saturation magnetization M_s
    ku = eh.bani2ku(20e-3, msat)  # [J/m^3] uniaxial anisotropy constant K_u
    delta = (90 - 20)*np.pi/180  # [rad] tilt of uniax. anisotropy from x axis
    bext = 5e-3  # [T] external magnetic inducton B_ext
    xi = (90 + 50) * np.pi / 180  # [rad] tilt of B_ext from x axis
    title = "Testing title for some composition of the plot."
    darling = eh.EpsminHeff(msat, name, loc, dpi, n, title=title,
                            demag_factors=demfs, ku=ku, delta=delta, bext=bext,
                            xi=xi)
    darling.use = [True, True, True]
    darling.plot = [True, True, True, True, True, True, True, False]
    # custom change of angle names from varphi to alpha
    darling.angles[2], darling.angles[3] = r"$\alpha$", r"$\alpha_0$"
    darling.evaluate()
    darling.title = "New custom title."
    darling.name = name + "_newtitle"  # new name not to overwrite the old plot
    darling.plot_heff()  # plot a new graph with new title


if __name__ == "__main__":
    main()
