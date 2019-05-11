from pathlib import Path

from inference import Inferrer

if __name__ == '__main__':
    data_root = Path("box-images")
    ng_samples = list(data_root.joinpath("NG").glob("*"))[:10]
    ok_samples = list(data_root.joinpath("OK").glob("*"))[:10]

    inferrer = Inferrer()

    for path in ok_samples:
        print(f
        "image { str(path) } is { inferrer(path) }")

        for path in ng_samples:
            print(f
            "image { str(path) } is { inferrer(path) }")
