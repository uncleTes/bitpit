# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
import numpy as np
cimport numpy as np

include "my_pablo_uniform.pyx"

cdef extern from *:
    ctypedef int D2 "2"

cdef extern from "My_Class_VTK.hpp" namespace "bitpit":
    cdef cppclass My_Class_VTK[G, D, dim]:
        My_Class_VTK(D* data_    ,
                     G& grid_    ,
                     string dir_ , 
                     string name_, 
                     string cod_ , 
                     int ncell_  , 
                     int npoints_, 
                     int nconn_) except +

        void printVTK()

        void AddData(string name_, 
                     int comp_   , 
                     string type_, 
                     string loc_ , 
                     string cod_)


cdef class Py_My_Class_VTK:
    cdef My_Class_VTK[MyPabloUniform,
                      double        ,
                      D2]* thisptr

    def __cinit__(self                                         , 
                  np.ndarray[double, ndim = 2, mode = "c"] data,
                  octree                                       ,
                  string directory                             ,
                  string file_name                             ,
                  string file_type                             ,
                  int n_cells                                  ,
                  int n_points                                 ,
                  int n_conn):
        self.thisptr = new My_Class_VTK[MyPabloUniform,
                                        double        ,
                                        D2](&data[0, 0]                                 ,
                                            (<Py_My_Pablo_Uniform>octree).der_thisptr[0],
                                            directory                                   ,
                                            file_name                                   ,
                                            file_type                                   ,
                                            n_cells                                     ,
                                            n_points                                    ,
                                            n_conn)

    def __dealloc__(self):
        del self.thisptr

    def print_vtk(self):
        self.thisptr.printVTK()

    def add_data(self              ,
                 string dataName   ,
                 int dataDim       ,
                 string dataType   ,
                 string pointOrCell,
                 string fileType):
        self.thisptr.AddData(dataName   ,
                             dataDim    ,
                             dataType   ,
                             pointOrCell,
                             fileType)

