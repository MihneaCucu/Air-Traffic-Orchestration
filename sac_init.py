"""
SAC Environment Setup
=====================

ACEST MODUL TREBUIE IMPORTAT PRIMUL în orice script care folosește SAC!

Setează variabilele de mediu necesare pentru a preveni PyTorch deadlock
pe macOS cu Python 3.13.

Usage:
------
import sac_init  # PRIMUL IMPORT - înainte de torch, numpy, etc!
import torch
from sac_agent import DiscreteSAC
...
"""

import os
import sys

# Set threading environment variables BEFORE any numpy/torch imports
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'sequential'

# Confirmation
_initialized = True

