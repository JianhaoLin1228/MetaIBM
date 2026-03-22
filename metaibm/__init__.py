"""MetaIBM core package.

Exports the four core classes for package-style imports, e.g.:
    from metaibm import individual, habitat, patch, metacommunity
"""

from .individual import individual
from .habitat import habitat
from .patch import patch
from .metacommunity import metacommunity

__all__ = ["individual", "habitat", "patch", "metacommunity"]
