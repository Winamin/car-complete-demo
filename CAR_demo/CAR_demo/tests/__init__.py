#!/usr/bin/env python3
"""
CAR Test Package
================

Contains all important tests:

Basic Tests:
    - test_car_comprehensive.py: Complete functionality test

Specialized Tests:
    - test_extreme_noise.py: Extreme noise robustness test
    - test_float128_limits.py: Float128 precision limit test
    - test_adversarial_attack.py: Adversarial attack test

Usage:
    from tests import test_car_comprehensive
    test_car_comprehensive.run_all_tests()
    
    # Or run directly
    python tests/test_extreme_noise.py
    python tests/test_float128_limits.py --limits
    python tests/test_adversarial_attack.py

Date: January 2026
"""

from .test_car_comprehensive import run_all_tests as run_comprehensive_tests

__all__ = [
    'run_comprehensive_tests',
]
