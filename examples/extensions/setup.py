# Copyright © 2023-2024 Apple Inc.

from tiki import extension
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="tiki_sample_extensions",
        version="0.0.0",
        description="Sample C++ and Metal extensions for Tiki primitives.",
        ext_modules=[extension.CMakeExtension("tiki_sample_extensions._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["tiki_sample_extensions"],
        package_data={"tiki_sample_extensions": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
        python_requires=">=3.10",
    )
