import numpy as np
import pickle
import exoplanet as xo
import pkg_resources

def angus_2019_model(log10_age, bprp):
    """
    Predicts rotation period from log10 color and log10 age.

    Only applicable to GK dwarfs.
    Args:
        log10_age (float): The (log10) age.
        bprp (float): The G_bp - G_rp color.
    Returns:
        log10_period (float): The period.
    """

    log10_bprp = np.log10(bprp)

    # Parameters with Solar bp - rp = 0.82
    p = [-38.957586198640314, 28.709418579540294, -4.919056437046026,
         0.7161114835620975, -4.716819674578521, 0.6470950862322454,
         -13.558898318835137, 0.9359250478865809]
    return 10**(np.polyval(p[:5], log10_bprp) + p[5]*log10_age)


def angus_2019_model_inverse(prot, bprp):
    """
    Predicts log10 age from log10 color and rotation period.

    Only applicable to GK dwarfs.
    Args:
        prot (array): The period array.
        bprp (array): The G_BP - G_RP color array.
    Returns:
        log10_age (array): The (log10) age in years.
    """
    log10_bprp = np.log10(bprp)
    log10_period = np.log10(prot)

    # Hard-code the gyro parameters
    p = [-38.957586198640314, 28.709418579540294, -4.919056437046026,
        0.7161114835620975, -4.716819674578521, 0.6470950862322454,
        -13.558898318835137, 0.9359250478865809]

    logage = (log10_period - np.polyval(p[:5], log10_bprp))/p[5]
    return logage


class GP_model(object):

    def __init__(self):
        # gp_model = pkg_resources.resource_filename(__name__, "gp_model.pkl")
        gp_model = pkg_resources.resource_filename(__name__,
                                                   "gp_model_01.27.21.pkl")
        with open(gp_model, "rb") as f:
            model, map_soln = pickle.load(f)

        with model:
            self.func = xo.get_theano_function_for_var(model.y_test)
            self.args = xo.utils.get_args_for_theano_function(map_soln)
            self.ind1 = model.vars.index(model.x1_test)
            self.ind2 = model.vars.index(model.x2_test)

    def pred_at(self, log10age, teff):
        """
        teff in K, log age in ln(age [Gyr]).
        """
        lnage = np.log((10**(log10age))*1e-9)
        self.args[self.ind1][0] = teff
        self.args[self.ind2][0] = lnage
        return np.exp(self.func(*self.args))
