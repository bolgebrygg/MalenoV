#!/usr/bin/env python
# -*- coding: utf 8 -*-
### ---- Functions for Input data(SEG-Y) formatting and reading ----
# Make a function that decompresses a segy-cube and creates a numpy array, and
# a dictionary with the specifications, like in-line range and time step length, etc.

from .csv_struct import csv_struct
from .segy_adder import segy_adder
from .segy_decomp import segy_decomp

__all__ = ["segy_adder",
           "segy_decomp",
           "csv_struct"
           ]
