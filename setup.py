from setuptools import setup, find_packages

setup(
    name='AcademicPython',
    version='0.1dev',
    url='https://github.com/chongchonghe/academic-python.git',
    author='Chong-Chong He',
    author_email='che1234@umd.edu',
    description='Some python functions and modules that are often used in my research.',
    #packages=find_packages(),
    packages=['academicpython'],
    install_requires=['astropy'],
)
