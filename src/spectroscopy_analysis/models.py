"""lmfit model classes: call them, e.g. ``gauss()``, to get a new, independent model."""

import functools
import lmfit


# ---- lmfit model classes
# These names are model CLASSES (or factories), not ready-made models: call
# them to get a new, independent model, e.g. ``gauss()`` or
# ``gauss(prefix='p1_')``. Every call gives a fresh object, so hints or
# prefixes set on one model never leak into another fit.
# (Before, they were single instances shared by the whole module, and
# set_param_hint() in one method changed later fits.)
gauss = lmfit.models.GaussianModel
lorentz = lmfit.models.LorentzianModel
voigt = lmfit.models.VoigtModel
baseline_line = lmfit.models.LinearModel
constant = lmfit.models.ConstantModel
poly_degree = 2
# PolynomialModel needs its degree: partial() fixes it, so baseline_poly()
# works like the other names
baseline_poly = functools.partial(lmfit.models.PolynomialModel, degree=poly_degree)
exp = lmfit.models.ExponentialModel


def as_model(model):
    """Return an lmfit model from a model class, a factory or an instance.

    Lets arguments such as ``bl`` or ``shape`` accept both the module names
    (``baseline_line``, ``voigt``, ...: classes) and a model created by the
    caller (``lmfit.models.QuadraticModel()``: an instance).

    - class or factory -> called, so a new model is returned;
    - instance         -> returned unchanged (the methods that use it do
                          not modify it).
    """
    if isinstance(model, lmfit.Model):
        return model
    return model()
