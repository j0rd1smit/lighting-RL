import os
from typing import List


def _load_requirements(path_dir: str, file_name: str, comment_char: str = "#") -> List[str]:
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


if __name__ == "__main__":
    _PATH_ROOT = os.path.dirname(__file__)
    print(_load_requirements(_PATH_ROOT, "requirements.txt"))
    print(_load_requirements(_PATH_ROOT, "requirements-dev.txt"))
