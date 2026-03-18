"""Bootstrap public datasets and caches."""

from pmtmax.cli.main import build_dataset

if __name__ == "__main__":
    build_dataset(cities=["Seoul", "NYC"])
