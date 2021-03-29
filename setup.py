import os

from setup_tools import _load_requirements
from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

_PATH_ROOT = os.path.dirname(__file__)


setup(
    name="lighting-rl",
    version="0.0.0",
    packages=find_packages(exclude=["tests", "examples"]),
    long_description_content_type="text/markdown",
    install_requires=_load_requirements(_PATH_ROOT, "requirements.txt"),
    extras_require={
        "dev": _load_requirements(_PATH_ROOT, "requirements-dev.txt"),
    },
    license="Creative Commons Attribution-Noncommercial-Share Alike license",
    long_description="",
)
