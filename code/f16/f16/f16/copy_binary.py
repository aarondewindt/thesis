from shutil import copyfile
from pathlib import Path

source_path = Path(__file__).parent.parent.parent / "libpyf16" / "target" / "debug" / "libpyf16.so"
destination_path = Path(__file__).parent / "libpyf16.so"

try:
    copyfile(source_path, destination_path)
except FileNotFoundError:
    print("libpyf16 binary not found. You might have to compile it.")
    pass
