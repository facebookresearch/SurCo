# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

setuptools.setup(
    name="dreamshard",
    version='1.0.0',
    author="Nobody",
    author_email="nobody@nobody",
    description="dreamshard",
    url="https://github.com/nobody/dreamshard",
    keywords=["Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    requires_python='>=3.8',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
