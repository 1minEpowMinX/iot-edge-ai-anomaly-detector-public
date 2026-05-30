"""Package entry point."""

import matplotlib

matplotlib.use("Agg")  # off-screen backend

from src.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
