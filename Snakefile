PROJECT = "eql_source_layouts"
BLACK_FILES = f"setup.py {PROJECT}".split()
FLAKE8_FILES = f"setup.py {PROJECT}".split()
LINT_FILES = f"setup.py {PROJECT}".split()


rule install:
    input:
        project = PROJECT,
        setup = "setup.py"
    shell:
        "pip install --no-deps -e ."


rule check:
    input:
        black = BLACK_FILES,
        flake8 = FLAKE8_FILES
    shell:
        "black --check {input.black} & "
        "flake8 {input.flake8}"


rule format:
    input:
        BLACK_FILES
    shell:
        "black {input}"


rule lint:
    input:
        LINT_FILES
    shell:
        "pylint --jobs=0 {input}"
