# -*- coding: utf-8 -*-
"""
Compatibility facade for the split implementation.

This module re-exports the four core classes so existing imports such as:
    import metacommunity_IBM as metaIBM
    from metacommunity_IBM import patch, metacommunity
continue to work without modifying model.py or mpi_running.py.
"""

from individual import individual
from habitat import habitat
from patch import patch
from metacommunity import metacommunity

__all__ = ["individual", "habitat", "patch", "metacommunity"]
