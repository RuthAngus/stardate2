"""
STARDATE2
=====================
This package provides a gyrochronology likelihood function.
"""

import numpy as np
from .model import *


def lnlike(log10_age, prot, prot_err, x, model):
    """
    The likelihood of the rotation period and color given the age.
    """

    # if model == "2019":
    #     model_prot = angus_2019_model(log10_age, x)
    # model_prot = gp.pred_at(log10_age, x)

    model_prot = model(log10_age, x)
    return -.5 * (model_prot - prot)**2/prot_err**2


def lnprior(log10_age):
    """
    The prior over age.
    """
    if log10_age < 20:
        return 0
    return -np.inf


def lnprob(log10_age, prot, prot_err, bprp, model):
    """
    The probability of the model given the data
    """
    return lnlike(log10_age, prot, prot_err, bprp, model) + lnprior(log10_age)
