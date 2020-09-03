"""
STARDATE2
=====================
This package provides a gyrochronology likelihood function.
"""

import numpy as np


def lnlike(log10_age, prot, prot_err, bprp):
    """
    The likelihood of the rotation period and color given the age.
    """

    model_prot = angus_2019_model(log10_age, bprp)
    return -.5 * (model_prot - prot)**2/prot_err**2


def lnprior(log10_age):
    """
    The prior over age.
    """
    if log10_age < 20:
        return 0
    return -np.inf


def lnprob(log10_age, prot, prot_err, bprp):
    """
    The probability of the model given the data
    """
    return lnlike(log10_age, prot, prot_err, bprp) + lnprior(log10_age)
