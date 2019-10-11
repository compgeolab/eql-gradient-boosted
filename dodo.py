"""
dodo file for installing, testing and linting the package
"""
import os
import shutil
from pathlib import Path


PROJECT = "eql_source_layouts"
SETUP_FILE = "setup.py"
TESTDIR = "tmp-test-dir-with-unique-name"
PYTEST_ARGS = (
    "--cov-config=../.coveragerc --cov-report=term-missing "
    + "--cov={} --doctest-modules -v --pyargs".format(PROJECT)
)
BLACK_FILES = [SETUP_FILE, PROJECT]
LINT_FILES = [SETUP_FILE, PROJECT]
FLAKE8_FILES = [SETUP_FILE, PROJECT]

DOIT_CONFIG = {"default_tasks": ["install", "check", "lint"], "verbosity": 2}


def task_install():
    "Install eql_source_layouts in editable mode"
    return {"actions": ["pip install -e ."]}


def task_test():
    "Run the test suite (including doctests) and report coverage"
    return {
        "actions": [
            f"mkdir -p {TESTDIR}",
            f"cd {TESTDIR}; MPLBACKEND='agg' pytest {PYTEST_ARGS} {PROJECT}",
            f"cp {TESTDIR}/.coverage* .",
            f"rm -rvf {TESTDIR}",
        ]
    }


def task_clean_all():
    "Clean up build and generated files"
    files_to_remove = [str(i) for i in Path(".").glob("**/*.pyc")]
    files_to_remove += [str(i) for i in Path(".").glob("**/*.coverage")]
    files_to_remove += [
        "build",
        "dist",
        "MANIFEST",
        "*.egg-info",
        "__pycache__",
        "coverage",
        ".cache",
        ".pytest_cache",
        "dask-worker-space",
    ]
    files_to_remove.append(TESTDIR)
    return {"actions": ["rm -rvf " + " ".join(files_to_remove)]}


def task_format():
    "Run black to automatically format the code"
    return {"actions": [run_black(BLACK_FILES)]}


def task_check():
    "Run code style and quality checks through black and flake8"
    return {"actions": [run_black(BLACK_FILES, check=True), run_flake8(FLAKE8_FILES)]}


def task_lint():
    "Lint code through pylint"
    return {"actions": [run_pylint(LINT_FILES)]}


def run_black(files, check=False):
    "Run black"
    command = ["black"]
    if check:
        command.append("--check")
    command.extend(files)
    return " ".join(command)


def run_flake8(files):
    "Run flake8"
    command = ["flake8"]
    command.extend(files)
    return " ".join(command)


def run_pylint(files):
    "Run pylint"
    command = ["pylint", "--jobs=0"]
    command.extend(files)
    return " ".join(command)
