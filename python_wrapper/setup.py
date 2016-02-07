# set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab
from Cython.Build import cythonize
from Cython.Distutils import Extension
# Change this commented line with the following one gives us the possibility
# to pass \"cython_compile_time_env\" at \"Extension\" because in line 154 we do
# not have more to call \"cythonize\" function before returning \"ext_modules\".
#from distutils.command.build_ext import build_ext as _build_ext
from Cython.Distutils import build_ext as _build_ext
from distutils.core import setup

from numpy import get_include as numpy_get_include

import imp
import os
import re
import sys

try:
    imp.find_module('mpi4py')
    ENABLE_MPI4PY = 1
except ImportError:
    ENABLE_MPI4PY = 0
    print("Dude, no module \"mpi4py\" found in your environment. We are " +
          "going serial.")

class build_ext(_build_ext):
    description = ("Custom build_ext \"BitPMesh-include-path\"" +
                   " \"mpi-include-path\""                      +
                   "\"BitPBase-include-path\""                  + 
                   " \"extension-source\" command for Cython")

    # Getting \"user_options\" from class \"build_ext\" imported as 
    # \"_build_ext\".
    user_options = _build_ext.user_options

    # Add new \"user_options\" \"BitP_Mesh-lib-path\", \"mpi-library-path\",
    # \"extensions-source\" and \"BitP_Base-lib-path\".
    user_options.append(("BitPMesh-include-path=", "B", "BitP_Mesh include path"))
    user_options.append(("BitPBase-include-path=", "Z", "BitP_Base include path"))
    user_options.append(("mpi-include-path=", "M", "mpi include path"))
    user_options.append(("extensions-source=", "E", "extensions source file"))

    def find_mpi_include_path(self):
    	LIBRARY_PATHS = os.environ.get("LD_LIBRARY_PATH")
    	mpi_checked = False
    	MPI_INCLUDE_PATH = None

    	for LIBRARY_PATH in LIBRARY_PATHS.split(":"):
            if (("mpi" in LIBRARY_PATH.lower()) and 
                (not mpi_checked)):
                MPI_INCLUDE_PATH = LIBRARY_PATH
                if not "/include/" in MPI_INCLUDE_PATH:
                    MPI_INCLUDE_PATH = re.sub("/lib?"    , 
                                              "/include/",
                                              MPI_INCLUDE_PATH)
                
                mpi_checked = True
                break
    
    	if (not mpi_checked):
            print("Dude, No \"mpi-include-path\" found in your " +
            	  "\"LD_LIBRARY_PATH\" environment variable. "   +
            	  "We are going serial.")
            
        return MPI_INCLUDE_PATH
	
    def find_BitP_Mesh_include_path(self):
        BITPMESH_INCLUDE_PATH = os.environ.get("BITPMESH_INCLUDE_PATH")

        if (BITPMESH_INCLUDE_PATH is None):
            print("Dude, no \"BITPMESH_INCLUDE_PATH\" environment " +
                  "variable found. Please, check this out or "      +
                  "enter it via shell.")

            sys.exit(1)

        return BITPMESH_INCLUDE_PATH
    
    def find_BitP_Base_include_path(self):
        BITPBASE_INCLUDE_PATH = os.environ.get("BITPBASE_INCLUDE_PATH")

        if (BITPBASE_INCLUDE_PATH is None):
            print("Dude, no \"BITPBASE_INCLUDE_PATH\" environment " +
                  "variable found. Please, check this out or "      +
                  "enter it via shell.")

            sys.exit(1)

        return BITPBASE_INCLUDE_PATH

    def check_extensions_source(self):
        if ((self.extensions_source is None) or 
            (not self.extensions_source.endswith(".pyx"))):
            print("Dude, insert source \".pyx\" file to build.")
	    
            sys.exit(1)
		
    def initialize_options(self):
        # Initialize father's \"user_options\".
        _build_ext.initialize_options(self)

        # Initializing own new \"user_options\".
        self.BitPMesh_include_path = None
        self.BitPBase_include_path = None
        self.mpi_include_path = None
        self.extensions_source = None

    def finalize_options(self):
        # Finalizing father's \"user_options\".
        _build_ext.finalize_options(self)
		
        # If yet \"None\", finalize own new \"user_options\" searching their
        # values.
        if (self.mpi_include_path is None):
            self.mpi_include_path = self.find_mpi_include_path()
        if (self.BitPMesh_include_path is None):
            self.BitPMesh_include_path = self.find_BitP_Mesh_include_path()
        if (self.BitPBase_include_path is None):
            self.BitPBase_include_path = self.find_BitP_Base_include_path()

        # Check if the source to pass at the \"Extension\" class is present and
	# finishes with \".pyx\".
	self.check_extensions_source()
	
        # Define \"custom cython\" extensions.
        self.extensions = self.def_ext_modules()

    # Define \"Extension\" being cythonized.
    def def_ext_modules(self):
        # Overloading compilers.
        os.environ["CXX"] = "c++"
        os.environ["CC"] = "gcc"
        ENABLE_MPI = 0
        include_libs = "-I" + self.BitPMesh_include_path
        include_libs = include_libs + " -I" + self.BitPBase_include_path

        if ((not (not self.mpi_include_path)) and (ENABLE_MPI4PY)):
            ENABLE_MPI = 1
            include_libs = include_libs + " -I" + self.mpi_include_path
            os.environ["CXX"] = "mpic++"
            os.environ["CC"] = "mpicc"

        _extra_compile_args = ["-std=c++11",
                               "-O3"       ,
                               "-fPIC"     ,
                               include_libs,
                               "-DENABLE_MPI=" + str(ENABLE_MPI)]
        _extra_link_args = ["-fPIC"] # Needed? We have already the same flag for 
                                     # the compiler args above.
        _cython_directives = {"boundscheck": False,
                              "wraparound": False}
        _language = "c++"
        _extra_objects = ["libbitpit_MPI.a" if (ENABLE_MPI) \
                                               else "libbitpit.a"]
        _include_dirs=["."                       , 
                       self.BitPMesh_include_path,
                       self.BitPBase_include_path,
                       numpy_get_include()       ,
                       "/home/federico/WorkSpace/PythonProjects/PhdThesis/bitpit/src/common",
                       "/home/federico/WorkSpace/PythonProjects/PhdThesis/bitpit/src/operators"]
        
        # Cython compile time environment.
        _cc_time_env = {"ENABLE_MPI": ENABLE_MPI}
	
        ext_modules = [Extension(os.path.splitext(self.extensions_source)[0],
                                 [self.extensions_source]                   ,              
                                 extra_compile_args = _extra_compile_args   ,
                                 extra_link_args = _extra_link_args         ,
        			 cython_directives = _cython_directives     , 
        			 language = _language                       ,
        			 extra_objects = _extra_objects             ,
        			 include_dirs = _include_dirs               ,
                                 cython_compile_time_env = _cc_time_env     , 
        			)]
        #return cythonize(ext_modules)
        return ext_modules

    def run(self):
        _build_ext.run(self)

setup(cmdclass = {"build_ext" : build_ext})
