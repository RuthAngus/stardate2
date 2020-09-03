import numpy as np


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
