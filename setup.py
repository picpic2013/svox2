from setuptools import setup
import os
import os.path as osp
import warnings

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('svox2_src/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []
include_dirs = [osp.join(ROOT_DIR, "svox2_src", "csrc", "include")]

# From PyTorch3D
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
	prefix = os.environ.get("CONDA_PREFIX", None)
	if prefix is not None and os.path.isdir(prefix + "/include/cub"):
		cub_home = prefix + "/include"

if cub_home is None:
	warnings.warn(
		"The environment variable `CUB_HOME` was not found."
		"Installation will fail if your system CUDA toolkit version is less than 11."
		"NVIDIA CUB can be downloaded "
		"from `https://github.com/NVIDIA/cub/releases`. You can unpack "
		"it to a location of your choice and set the environment variable "
		"`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
	)
else:
	include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))

try:

    def isFileCuOrCpp(fileName, postfixs):
        for post in postfixs:
            if fileName[-len(post) - 1:] == '.' + post:
                return True
        return False

    def addFileToCsrcList(basePath, srcList, postfixs=None):
        if postfixs is None:
            postfixs = ['cu', 'cpp']
        commonFileList = os.listdir(os.path.join(basePath))
        commonFileList = [os.path.join(basePath, _) for _ in commonFileList if isFileCuOrCpp(_, postfixs)]
        srcList.extend(commonFileList)

    basePath = os.path.join('svox2_src', 'csrc')
    csrcList = []

    addFileToCsrcList(basePath, csrcList)
    addFileToCsrcList(os.path.join(basePath, 'pic_cuvol'), csrcList)

    ext_modules = [
        CUDAExtension(
            'svox2.csrc', 
            csrcList, 
            include_dirs=include_dirs,
            optional=False
        ),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='svox2',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='PyTorch sparse voxel volume extension, including custom CUDA kernels',
    long_description='PyTorch sparse voxel volume extension, including custom CUDA kernels',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['svox2', 'svox2.csrc'],
    package_dir={
        'svox2': 'svox2_src'
    }, 
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
