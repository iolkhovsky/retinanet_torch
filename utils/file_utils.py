from glob import glob
from os.path import join, splitext


def get_files_list(root):
    return glob(join(root, "*.*"))


def get_imgs_list(root, valid_ext=[".jpg", ".png"]):
    return [x for x in get_files_list(root) if splitext(x)[1] in valid_ext]
