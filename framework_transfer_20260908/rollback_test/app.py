"""Project-root launcher for local development."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anti_fraud_explorer.api import main  # noqa: E402


if __name__ == "__main__":
    main()
