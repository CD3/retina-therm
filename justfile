set positional-arguments := true

default:
    just --list

test *args:
    cd tests && uv run pytest -s {{ args }}

publish:
    rm -rf dist
    uv build
    uv publish
