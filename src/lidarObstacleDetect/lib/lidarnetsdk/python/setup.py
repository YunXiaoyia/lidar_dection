
import os
import re
import subprocess
import sys
import shutil
from typing import List, Tuple
from pathlib import Path
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext

_LDD_ARROW_OUTPUT_RE = re.compile(
    r"(?P<soname>.+)\s=>\s(?P<dep_path>.*)\s\(?(?P<mem_address>\w*)\)?")
_LDD_NON_ARROW_OUTPUT_RE = re.compile(
    r"(?P<dep_path>.+)\s\(?(?P<mem_address>\w*)\)?")


def parse_ldd_output(cmd_out):
    ret: List[Tuple[str, Path, str]] = []
    lines = cmd_out.decode().split('\n')
    for line in lines[:-1]:
        line = line.strip('\t')
        if '=>' in line:
            mtch = _LDD_ARROW_OUTPUT_RE.match(line)
            if not mtch:
                raise RuntimeError(
                    ("Unexpected ldd output. Expected to match {}, "
                     "but got: {!r}").format(_LDD_ARROW_OUTPUT_RE.pattern,
                                             line))
            ret.append(
                (mtch['soname'], Path(mtch['dep_path']), mtch['mem_address']))

        else:
            mtch = _LDD_NON_ARROW_OUTPUT_RE.match(line)
            if not mtch:
                raise RuntimeError(
                    ("Unexpected ldd output. Expected to match {}, "
                     "but got: {!r}").format(_LDD_NON_ARROW_OUTPUT_RE.pattern,
                                             line))
    return ret


NV_FAMILIES = [
    "nvinfer",
    "cudart",
    "cudnn",
    "cublas",
    "cufft",
    "nvrtc",
    "myelin",
    "nvparser",
    "nvcaffeparser",
    "nvonnxparser",
]

item_in_nvfamily = lambda item: any(i in item for i in NV_FAMILIES)


def fetch_nv_so(ldd_outputs) -> List[str]:
    ret = [str(item[1]) for item in ldd_outputs if item_in_nvfamily(item[0])]
    return ret


def cp_nv_so(nv_so, target_dir):
    for item in nv_so:
        item_resolve = str(
            Path(item).resolve()) if Path(item).is_symlink() else item
        shutil.copy(item_resolve, Path(target_dir) / Path(item).name)


# ldd to check output library
trtexec_path = subprocess.check_output([
    "which",
    "trtexec",
]).decode().strip("\n")

if not Path(trtexec_path).exists():
    raise Exception("trtexec required")

pillarx_libs = [
    str(Path("/opt/pillarx_sdk/lib/libpillarx-sdk.so").absolute()),
    str(Path("/opt/pillarx_sdk/lib/libtrtexec_plugins.so").absolute()),
    str(Path(trtexec_path).absolute()),
]

all_so = set(pillarx_libs)
for lib in pillarx_libs:
    ldd_pillarx_libs = subprocess.check_output(["ldd", lib])
    ldd_pillarx_so = parse_ldd_output(ldd_pillarx_libs)
    ldd_pillarx_nv_so = fetch_nv_so(ldd_pillarx_so)
    all_so.update(ldd_pillarx_nv_so)
print(all_so)
cp_nv_so(all_so, "pillarx")

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = str(Path(sourcedir).absolute())


class CMakeBuild(build_ext):

    def build_extension(self, ext):
        extdir = str(Path(self.get_ext_fullpath(ext.name)).absolute())
        print(f"ext = {ext}; ext.name = {ext.name}")
        print(f"ext.sourcedir {ext.sourcedir}")

        # required for auto-detection & inclusion of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        self.debug = True
        debug = int(os.environ.get("DEBUG",
                                   0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        extdir = Path(extdir).parent / "pillarx"
        print("extdir", extdir)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython3_ROOT_DIR={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [
                item for item in os.environ["CMAKE_ARGS"].split(" ") if item
            ]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"
        ]

        import pybind11
        cmake_args += [
            f"-DPYBIND11_INCLUDE_PATH={pybind11.get_include()}",
            f"-DPYBIND11_PACKAGE_PATH={pybind11.get_cmake_dir()}"
        ]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator:
                try:
                    import ninja  # noqa: F401

                    cmake_args += ["-GNinja"]
                except ImportError:
                    pass

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator
                                for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [
                    "-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))
                ]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            self.parallel = os.cpu_count() - 1
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # additional cmake_args
        print(" ".join(["cmake", ext.sourcedir] + cmake_args))
        print(" ".join(["cmake", "--build", "."] + build_args))

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args,
                              cwd=self.build_temp)


setup(
    name="PillarX",
    version="0.2.3",
    description="PillarX Inference SDK",
    long_description="PillarX Inference SDK",
    author="NVIDIA",
    license="Proprietary",
    classifiers=[
        "License :: Other/Proprietary License",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    ext_modules=[CMakeExtension("pillarx")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=[
        'pybind11',
    ],
    setup_requires=['pybind11'],
    extras_require={"numpy": "numpy"},
    package_data={"pillarx": ["*.so*", "trtexec"]},
    include_package_data=True,
    zip_safe=True,
)
