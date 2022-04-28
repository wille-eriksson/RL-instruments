import pathlib
import pkg_resources
from setuptools import setup, find_packages

with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setup(
    name='rl_instruments',
    description='Reinforcement Learning for Musical Instrument Models Exploration',
    author='Wille Eriksson',
    author_email='wille.eriksson96@gmail.com',
    packages=find_packages(),
    install_requires=install_requires
)
