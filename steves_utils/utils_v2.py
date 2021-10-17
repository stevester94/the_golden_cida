#! /usr/bin/python3

import os

def get_datasets_base_path():
    return os.environ["DATASETS_ROOT_PATH"]

def get_files_with_suffix_in_dir(path, suffix):
    """Returns full path"""
    (_, _, filenames) = next(os.walk(path))
    return [os.path.join(path,f) for f in filenames if f.endswith(suffix)]