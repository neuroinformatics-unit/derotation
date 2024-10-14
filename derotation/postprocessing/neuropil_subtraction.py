import vendor.allensdk.brain_observatory.dff as dff_module
from vendor.allensdk.brain_observatory.r_neuropil import NeuropilSubtract


def neuropil_subtraction(f, f_neu) -> tuple:
    """Compute neuropil subtraction and dF/F for a given ROI

    Parameters
    ----------
    f : np.ndarray
        Fluorescence trace of the ROI
    f_neu : np.ndarray
        Fluorescence trace of the neuropil

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        dF/F and neuropil signal
    """
    neuropil_subtraction = NeuropilSubtract()
    neuropil_subtraction.set_F(f, f_neu)
    neuropil_subtraction.fit()

    r = neuropil_subtraction.r

    f_corr = f - r * f_neu

    # kernel values to be chossen for 3-photon data
    dff = 100 * dff_module.compute_dff_windowed_median(
        f_corr, median_kernel_long=1213, median_kernel_short=23
    )

    return dff, r
