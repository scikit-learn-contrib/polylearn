from __future__ import print_function
import os.path
import sys
import setuptools
from numpy.distutils.core import setup


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


DISTNAME = 'polylearn'
DESCRIPTION = ("Factorization machines and polynomial networks "
               "for classification and regression in Python.")
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Vlad Niculae'
MAINTAINER_EMAIL = 'vlad@vene.ro'
URL = 'https://contrib.scikit-learn.org/polylearn'
LICENSE = 'Simplified BSD'
DOWNLOAD_URL = 'https://github.com/scikit-learn-contrib/polylearn'
VERSION = '0.1.dev0'


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)

    config.add_subpackage('polylearn')

    return config


if __name__ == '__main__':
    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          install_requires=[
              'six',
              'scikit-learn'
          ],
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers', 'License :: OSI Approved',
              'Programming Language :: C', 'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX', 'Operating System :: Unix',
              'Operating System :: MacOS'
             ]
          )