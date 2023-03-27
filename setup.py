# from setuptools import find_packages, setup

# setup(
#     name='gsort',
#     packages=find_packages(),
# )


from setuptools import find_packages#, setup
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize



ext_modules = [
    Extension("gsort.utilities.cython_extensions.bin2py_cythonext", 
                ["gsort/utilities/cython_extensions/bin2py_cythonext.pyx",]),
    Extension("gsort.utilities.cython_extensions.visionfile_cext",
                    ["gsort/utilities/cython_extensions/visionfile_cext.pyx", ]),
    Extension("gsort.utilities.cython_extensions.visionwrite_cext",
                ["gsort/utilities/cython_extensions/visionwrite_cext.pyx",]),
]


setup(
    name='gsort',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
#     cmd_class={'build_ext' : build_ext}
)



# distutils.core.setup(

#         )
