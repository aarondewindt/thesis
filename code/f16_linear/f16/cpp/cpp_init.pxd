from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.string cimport string
from libc.stdint cimport uint64_t

ctypedef vector[double]* dvec_ptr
ctypedef map[string, dvec_ptr]* dvec_smap_ptr 
