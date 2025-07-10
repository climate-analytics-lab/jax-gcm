#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    use_scm_version={"fallback_version": "999"},
    packages=find_packages()
)