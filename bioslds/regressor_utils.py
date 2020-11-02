""" Define various utilities for use with regressors. """

import numpy as np

from bioslds.utils import to_hankel

from typing import Sequence, Any


def transform_ar(regressor: Any, y: Sequence, bias: bool = False, **kwargs) -> Any:
    """ Use a regression model to infer autoregressive parameters.

    This is a convenience function that uses `utils.to_hankel` to generate values for
    the predictor variables (effectively converting them into lag vectors) and then
    calls `regressor.transform` to fit a regression model and infer the underlying
    latent trajectory. This turns the regression model into an autoregressive one. If
    `bias` is true, the predictor variables are augmented with a constant predictor,
    equal to 1 at all time points, allowing to fit autoregressive processes with
    non-zero mean.

    The variables are ordered such that the regression coefficients `w[i]` obey one of
    the following equations:
        no bias:    y[t] \approx sum(w[i] * y[t - i - 1] for i in range(p))
        with bias:  y[t] \approx sum(w[i] * y[t - i - 1] for i in range(p)) + w[p]
    where `p` is the order of the autoregressive process, which is taken to be:
        no bias:    p = regressor.n_features
        with bias:  p = regressor.n_features - 1

    Lag vectors cannot be calculated for the first `p` samples of the input, and so the
    length of the output will be `len(y) - p` instead of the `len(y)` samples typically
    returned by `regressor.transform`.

    Parameters
    ----------
    regressor
        Regression object to use for fitting. This should have a method `transform` like
        the classes in `bioslds.regressors`.
    y
        Sequence of values for the dependent variable. This is used to generate lag
        vectors that are then used as predictors.
    bias
        If true, a constant predictor is added, to model autoregressive processes with
        non-zero mean
    All other keyword arguments are passed directly to `regressor.transform`.

    Returns the output from `regressor.transform(Xar, yar)` called with `yar = y[p:]`
    and the matrix `Xar` of calculated lag vectors. Note that the first `p` samples of
    `y` don't have a well-defined lag vector, and so no inference is run for these
    samples (though they are used for the inference of subsequent samples).
    """
    p = regressor.n_features - (1 if bias else 0)
    X = to_hankel(y, p)

    if bias:
        X = np.hstack((X, np.ones((len(X), 1))))

    return regressor.transform(X[p - 1: -1], y[p:], **kwargs)
