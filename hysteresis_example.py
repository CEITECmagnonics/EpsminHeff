import numpy as np
# import matplotlib.cm as mcm  # unblock for matplotlib colormaps
import epsmin_heff as eh
from cmcrameri import cm  # see https://www.fabiocrameri.ch/colourmaps/


def main():
    """Script for hysteresis-loop modelling using the epsmin_heff module."""
    # saving name w/o extension
    name = "SW_00_xi90_delta75_Bani6p5mT_x25um_y5um_z10nm"
    # saving directory
    loc = "hystloop_example_plots"
    dpi = 250  # DPI resolution of the plots
    n, loopn = 200, 30  # number of points in energy profiles and in the loop
    rectdim = (25e-6, 5e-6, 10e-9)  # [um] sample dimensions
    # demagnetisation factors of a rectangular prism (Aharoni model)
    demfs = (eh.aharoni(rectdim[1], rectdim[2], rectdim[0]),
             eh.aharoni(rectdim[2], rectdim[0], rectdim[1]),
             eh.aharoni(*rectdim))
    msat = 830e3  # [A/m] saturation magnetization M_s
    ku = eh.bani2ku(6.5e-3, msat)  # [J/m^3] uniaxial anisotropy constant K_u
    delta = 75 * np.pi / 180  # [rad] tilt of uniax. anisotropy from x axis
    bext = 25e-3  # [T] maximal external magnetic inducton B_ext
    xi = 90 * np.pi / 180  # [rad] tilt of B_ext from x axis

    # Set base EpsminHeff object with all needed energies, maximal B_ext
    # will be taken from here, as well as some other parameters.
    ehbase = eh.EpsminHeff(msat, name, loc, dpi, n, True, True, True, True,
                           True, demag_factors=demfs, ku=ku, delta=delta,
                           bext=bext, xi=xi, disp_messages=False)
    # Create a Hysteresis object and plot energy profiles with a sequential
    # cmap, here I use a perceptually uniform colormap cm.batlow.
    hlbase = eh.Hysteresis(ehbase, loopn, None, True, True, cm.batlow, True)
    hlbase.evaluate()
    # change colormap to categorical (or qualitative) type, e.g. "tab20", but
    # since matplotlib has only categorical colormaps with 20 different colors,
    # I will use cmcrameri.cm.lapazS
    # hlbase.edp_cmap = mcm.get_cmap("tab20")  # example for matplotlib cmaps
    hlbase.edp_cmap = cm.lapazS
    hlbase.edp_cmaptype = "categorical"
    hlbase.ehobj.name = name+"_cat"
    hlbase.plot_edp()  # Since no data parameters were changed, I do not need
    # ...to rerun the calculations, just plot again. (There is an exception,
    # ...when I would need to call the calculations again, that is when I'd
    # ...change the angle labels in hlbase.ehobj and want to change the axis
    # ...labels in hlbase that way.)
    # now let's change some parameters and run the process again
    name = name[:4] + "1" + name[5:]  # changes index in the name from 00 to 01
    hlbase.ehobj.name = name
    hlbase.mess = False  # now without status messages
    hlbase.loopn = 100  # increase the number of loop points
    hlbase.reset_computation()  # re-initialize main calculation variables
    hlbase.evaluate()


if __name__ == "__main__":
    main()
