import os.path

import numpy


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('polylearn', parent_package, top_path)

    config.add_extension('loss_fast', sources=['loss_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('cd_direct_fast', sources=['cd_direct_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('cd_linear_fast', sources=['cd_linear_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    config.add_extension('cd_lifted_fast', sources=['cd_lifted_fast.cpp'],
                         include_dirs=[numpy.get_include()])

    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
