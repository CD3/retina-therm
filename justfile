set positional-arguments

default:
  just --list

test:
  nox

pytest *args:
  cd tests && uv run pytest -s {{args}}

publish:
  rm -rf dist
  uv build
  uv publish
