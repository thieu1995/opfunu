from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="opfunu",
    version="0.7.1",
    author="Thieu Nguyen",
    author_email="nguyenthieu2102@gmail.com",
    description="A library of unconstrained Optimization Functions using Numpy (OpFuNu) for optimization problems",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thieunguyen5991/opfunu",
    download_url="https://github.com/thieunguyen5991/opfunu/archive/v0.7.1.zip",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: System :: Benchmark",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],
    install_requires=["numpy"],
    python_requires='>=3.6',
)