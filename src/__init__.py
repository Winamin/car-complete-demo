#!/usr/bin/env python3
"""
CAR: Cognitive Architecture with Retrieval-Based Learning
Main Source Code Package

This package provides a complete implementation of the CAR architecture
for extreme noise recognition research.

Date: January 2026
"""

from .config import CARConfig
from .knowledge_base import KnowledgeBase, KnowledgePattern
from .unit import ComputationalUnit, MultiViewAnalyzer
from .car_model import CompleteCARModel

__version__ = "1.0.0"
__author__ = "Yingxu Wang"

__all__ = [
    'CARConfig',
    'KnowledgeBase', 
    'KnowledgePattern',
    'ComputationalUnit',
    'MultiViewAnalyzer',
    'CompleteCARModel',
]
