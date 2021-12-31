#!/usr/bin/env python
# -*- coding: utf-8 -*-

import runpy
import os
import pytest
import glob

pytestmark = pytest.mark.scripts


THIS_FILES_DIR_PATH = os.path.realpath(os.path.dirname(__file__))


def get_paths_of_scripts():
    exclude_sub_strings = ["do_not_execute"]
    plot_script_paths = glob.glob(
        os.path.join(os.path.dirname(THIS_FILES_DIR_PATH), "examples", "*.py")
    )
    plot_script_paths_sorted = sorted(plot_script_paths)
    plot_script_paths_sorted_reduced = [
        p
        for p in plot_script_paths_sorted
        if not any(sub in p for sub in exclude_sub_strings)
    ]
    return plot_script_paths_sorted_reduced


class Test_scripts:
    @pytest.mark.parametrize(
        "path_script",
        get_paths_of_scripts(),
    )
    def test_execute_scripts(self, path_script):

        print(f"Execut script:\n{path_script}")
        runpy.run_path(path_script, init_globals={}, run_name="__main__")
