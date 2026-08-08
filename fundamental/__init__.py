"""Fundamental equity sleeve research and monitoring package.

The package is intentionally research-only in its first phase.  It produces
point-in-time source archives, deterministic candidate triage, and human
review artifacts; it does not construct or transmit broker commands.
"""

from .config import POLICY_VERSION, SLEEVE_POLICY

__all__ = ["POLICY_VERSION", "SLEEVE_POLICY"]

