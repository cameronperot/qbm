import setuptools

with open("requirements.txt", "r") as requirements_file:
    requirements = requirements_file.read().splitlines()

setuptools.setup(install_requires=requirements)
