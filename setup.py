import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src" / "dwn_cpu"

if os.name == "nt":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3"]
    if os.environ.get("DWN_CPU_NATIVE", "0") == "1":
        compile_args.append("-march=native")

ext_modules = [
    CppExtension(
        name="dwn_cpu._efd_fused_general_cpu",
        sources=["src/dwn_cpu/efd_fused_general_cpu.cpp"],
        extra_compile_args=compile_args,
    ),
    CppExtension(
        name="dwn_cpu._efd_fused_multilayer_cpu",
        sources=["src/dwn_cpu/efd_fused_multilayer_cpu.cpp"],
        extra_compile_args=compile_args,
    ),
]

setup(
    name="dwn_cpu",
    version="0.6.0",
    description="CPU training and inference kernels for Differentiable Weightless Neural Networks",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/alanbacellar/dwn_cpu",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={"dwn_cpu": ["*.cpp"]},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    install_requires=[
        "torch",
        "numpy"
    ],
    extras_require={
        "examples": ["numpy", "openml", "scikit-learn"],
    },
    python_requires=">=3.9",
    zip_safe=False,
)
