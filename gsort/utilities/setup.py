# from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# from Cython.Build import cythonize

# ext_modules = [
#     Extension("cython_extensions.bin2py_cythonext", 
#                 ["cython_extensions/bin2py_cythonext.pyx",]),
#     Extension("cython_extensions.visionfile_cext",
#                     ["cython_extensions/visionfile_cext.pyx", ]),
#     Extension("cython_extensions.visionwrite_cext",
#                 ["cython_extensions/visionwrite_cext.pyx",]),
# ]

# setup(
#     ext_modules=cythonize(ext_modules),
#     cmd_class={'build_ext' : build_ext}
#         )
