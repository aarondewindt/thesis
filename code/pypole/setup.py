from setuptools import setup, find_packages
from distutils.util import convert_path


ver_path = convert_path('pypole/version.py')
with open(ver_path) as ver_file:
    ns = {}
    exec(ver_file.read(), ns)
    version = ns['version']

setup(
    name='pypole',
    version=ns['version'],
    description="",
    author='Aaron de Windt',
    author_email='',
    url='',

    install_requires=['numpy',
                      'scipy',
                      "matplotlib",
                      "xarray",
                      "pandas",
                      "numba>=0.46.0"],
    packages=find_packages('.', exclude=["test"]),
    package_data={},
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha'],
    entry_points={}
)
