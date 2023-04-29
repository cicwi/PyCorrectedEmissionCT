#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(packages=find_packages(include=["corrct", "corrct.*"]))
