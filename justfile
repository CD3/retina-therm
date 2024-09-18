set positional-arguments

default:
  just --list

test:
  nox

pytest *args:
  cd tests && rye run pytest -s "$@"

