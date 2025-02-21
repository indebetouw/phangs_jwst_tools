# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = []
from .complim import *   # noqa
# Then you can be explicit to control what ends up in the namespace,
__all__ += ['fake_image', 'phot_catalog', 'completeness_analysis', 'completeness_limit']   # noqa
# or you can keep everything from the subpackage with the following instead
# __all__ += example_mod.__all__
