"""
configuration for pytest: setup and cleanup of test folders

author: Erick Alvarez
date: July 19 2024
"""
import os
from constants import EDA_IMGS_TEST_PTH, RES_IMGS_TEST_PTH, MODELS_TEST_PTH

test_dirs_pths = [
    EDA_IMGS_TEST_PTH,
    RES_IMGS_TEST_PTH,
    MODELS_TEST_PTH
]


def pytest_configure():
    """
        pytest configuration, running before all tests
    """
    setup()


def pytest_unconfigure():
    """
        pytest undo configuration, running after all tests
    """
    cleanup()


def remove_files_in_dir(dir_pth):
    """
        remove all file in the give directory if it exists
    """
    if os.path.exists(dir_pth):
        for file in os.listdir(dir_pth):
            os.remove(os.path.join(dir_pth, file))


def create_dir(dir_pth):
    """
        create a directory if it does not exist
    """
    os.makedirs(dir_pth, exist_ok=True)


def remove_dir(dir_pth):
    """
        removes directory if it exists
    """
    if os.path.exists(dir_pth):
        os.rmdir(dir_pth)


def setup():
    """
        remove any existent file from previous run from test directories
        create test directories
    """
    for test_dir_pth in test_dirs_pths:
        remove_files_in_dir(test_dir_pth)
        create_dir(test_dir_pth)


def cleanup():
    """
        remove created file during test
        remove test directories
    """
    for test_dir_pth in test_dirs_pths:
        remove_files_in_dir(test_dir_pth)
        remove_dir(test_dir_pth)
