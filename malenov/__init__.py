#!/usr/bin/env python
# -*- coding: utf 8 -*-
"""
### Function for n-dimensional seismic facies training /classification using Convolutional Neural Nets (CNN)
### By: Charles Rutherford Ildstad (University of Trondheim), as part of a summer intern project in ConocoPhillips and private work
### Contributions from Anders U. Waldeland (University of Oslo), Chris Olsen (ConocoPhillips), Doug Hakkarinen (ConocoPhillips)
### Date: 26.10.2017
### For: ConocoPhillips, Norway,
### GNU V3.0 lesser license
"""
from . import augment
from . import plotting
from . import predict
from . import segy
from . import train
from .malenov import master

__all__ = ["augment",
           "malenov",
           "plotting",
           "predict",
           "segy",
           "train"
           ]
