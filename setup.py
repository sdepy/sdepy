import os
import setuptools
from sdepy import __version__ as version

# project root directory
DIR = os.path.dirname(__file__)

# get readme file
with open(os.path.join(DIR, 'README.rst')) as f:
    readme = f.read()

if __name__ == '__main__':
    setuptools.setup(
        name='sdepy',
        version=version,
        description='SdePy: Numerical Integration of Ito '
        'Stochastic Differential Equations',
        long_description=readme,
        long_description_content_type='text/x-rst',
        url='https://github.com/sdepy/sdepy',
        download_url='https://pypi.org/project/sdepy',
        author='Maurizio Cipollina',
        author_email='sdepydev@gmail.com',
        license='BSD',
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: OS Independent',
            ],
        platforms=[
            'Windows',
            'Linux',
            'Solaris',
            'Mac OS-X',
            'Unix',
            ],
        python_requires='>=3.5',
        setup_requires=['numpy>=1.15.2', 'scipy>=0.19.1', 'wheel'],
        install_requires=['numpy>=1.15.2', 'scipy>=0.19.1'],
        packages=[
            'sdepy',
            'sdepy.tests',
            ],
        package_data={'sdepy': [
            'tests/_pytest.ini',
            'tests/cfr/*err_expected.txt',
            ]}
        )
