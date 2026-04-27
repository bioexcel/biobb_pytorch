"""Compatibility shim for legacy imports.

Allows importing ``biobb_pytorch.mdae.LRP`` while the implementation
lives in ``biobb_pytorch.mdae.explainability.LRP``.
"""

from .explainability.LRP import LRP, relevance_propagation, main
