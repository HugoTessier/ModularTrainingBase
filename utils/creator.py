import importlib.util
import os
import sys


def create(path):
    '''
    Loads a Python script at path and retrives its attribute called OBJECT. This attribute can be anything.

    :param path: Path to the script to load.
    :return: The OBJECT attribute, whatever it is.
    '''
    sys.path.append(path)
    sys.path.append(os.path.dirname(path))
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['module'] = module
    spec.loader.exec_module(module)
    return getattr(module, 'OBJECT')
