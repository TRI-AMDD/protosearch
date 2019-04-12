__maintainer__ = "Kirsten Winther"

import os

def get_basepath(calculator='vasp',
                 tri_path=None,
                 username=None,
                 ext=None):

    TRI_PATH = tri_path or os.environ['TRI_PATH']
    username = username or os.environ['TRI_USERNAME']
    basepath = TRI_PATH + '/model/{}/1/u/{}'\
        .format(calculator, username)
    if ext:
        basepath += '/{}'.format(ext)
        if not os.path.isdir(basepath):
            os.mkdir(basepath)

    return basepath
