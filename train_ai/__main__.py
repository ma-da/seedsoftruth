"""Allow ``python -m seedsoftruth.train_ai`` to invoke the trainer CLI."""

import sys

from .train import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
