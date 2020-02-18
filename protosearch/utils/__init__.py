__maintainer__ = "Kirsten Winther"

import os

from protosearch import __version__ as version


def get_tri_basepath(calculator='vasp',
                 tri_path=None,
                 username=None,
                 ext=None):

    TRI_PATH = tri_path or os.environ['TRI_PATH']
    username = username or os.environ['TRI_USERNAME']

    # Protosearch version identification
    protosearch_id = 'protosearch' + version.replace('.', '')

    basepath = TRI_PATH + '/model/{}/1/u/{}/{}'\
        .format(calculator, username, protosearch_id)
    if not os.path.isdir(basepath):
        os.mkdir(basepath)
    if ext:
        basepath += '/{}'.format(ext)
        if not os.path.isdir(basepath):
            os.mkdir(basepath)

    return basepath
