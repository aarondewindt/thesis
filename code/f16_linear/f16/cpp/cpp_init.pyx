from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cython.operator import dereference, postincrement
import xarray as xr
from itertools import product
import numpy as np


