"""The Python-installed CLI invokes the exact same CLI as the npm package."""
import os
import sys
from . import _node, _resources


def main() -> None:
    os.environ.setdefault("WAVEPARSE_PYTHON", sys.executable)
    node = _node()
    os.execv(node, [node, str(_resources() / "cli.cjs"), *sys.argv[1:]])


if __name__ == "__main__":
    main()
