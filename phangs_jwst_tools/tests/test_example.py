

def test_primes():
    # from ..complim import constrained_diffusion
    assert 3 == 3


def test_deprecation():
    import warnings
    warnings.warn(
        "This is deprecated, but shouldn't raise an exception, unless "
        "you ask pytest to turn warnings into errors.",
        DeprecationWarning)
