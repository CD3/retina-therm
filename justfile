set positional-arguments

default:
  just --list

test:
  nox

light-test *args:
  cd {{ justfile_directory()/"tests" }} && poetry run pytest -s "$@"

light-test-on-changes *args:
  cd {{ justfile_directory() }} && watchexec -e py just light-test -- "$@"

build-package:
  cd {{ justfile_directory()/"tests" }} && poetry build

upload-package:
  cd {{ justfile_directory()/"tests" }} && poetry publish
