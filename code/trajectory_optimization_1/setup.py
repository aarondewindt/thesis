from setuptools import setup, find_packages
from distutils.util import convert_path


ver_path = convert_path('traj1/version.py')
with open(ver_path) as ver_file:
    ns = {}
    exec(ver_file.read(), ns)
    version = ns['version']

setup(
    name='traj1',
    version=ns['version'],
    description="2D ground to orbit optimization using reinforcement learning.",
    author='Aaron de Windt',
    author_email='aaron.dewindt@gmail.com',
    url='https://github.com/aarondewindt/thesis',
    install_requires=[],
    packages=find_packages('.'),
    package_data={},
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Development Status :: 2 - Pre-Alpha'],
    entry_points={
        'console_scripts': []
    }
)
