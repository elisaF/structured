# distutils: language = c++
# distutils: sources = MaxSpanTree.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "MaxSpanTree.h" namespace "trees":
  cdef cppclass MaxSpanTree:
        MaxSpanTree() except +
        vector[vector[double]] get_tree(vector[vector[double]])

# creating a cython wrapper class
cdef class PyMaxSpanTree:
    cdef MaxSpanTree *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new MaxSpanTree()
    def __dealloc__(self):
        del self.thisptr
    def get_tree(self, str_scores):
        return self.thisptr.get_tree(str_scores)
