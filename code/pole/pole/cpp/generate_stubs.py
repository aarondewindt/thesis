import re
from pathlib import Path


pyx_path = Path(__file__).parent / "cpp_init.pyx"
stubs_path = pyx_path.parent / "cpp_init.pyi"
# pattern = r'(?:cdef class )(.+)(?::)|((?:    def )[\s\S]+?(?::))(?:(\s*"""[\s\S]*?""")|\s*\n)'
pattern = r'(?:cdef class )(.+)(?::)|((?:    def )[\s\S]+?(?::))(?:(\s*"""[\s\S]*?""")|\s*\n)|(    @[\s\S]*?\n)'
# pattern = r'(?:cdef class )(.+)(?::)|((?:\s*def )[\s\S]+?(?::))(?:(\s*"""[\s\S]*?""")|\s*\n)|(    @[\s\S]*?\n)'
# func_pattern = r"(?:\s*?def\s+?)(\w+?)(?:\()(.+?)(?:\):)"
func_pattern = r"(?:    def\s+?)(\w+?)(?:\()([\s\S]+?)(?:\))([\w\W]*?)(?::)"
param_pattern = r"(?:\s*?\w+\s+)?([\w\W]+?)$"


with open(pyx_path) as pyx_file:
    with open(stubs_path, "w") as stubs_file:
        stubs_file.write("from typing import Sequence, Tuple, Dict, Optional, Union, Iterable\nimport numpy as np\nfrom os import PathLike\n\n")

        for match in re.finditer(pattern, pyx_file.read()):
            if match.group(1) is not None:
                stubs_file.write(f"\nclass {match.group(1)}:\n")
            elif match.group(2) is not None:
                match_2 = match.group(2).replace("__cinit__", "__init__")

                func_match = re.match(func_pattern, match_2)
                if func_match is None:
                    continue
                func_name = func_match.group(1)
                func_params = func_match.group(2)
                func_params = [re.match(param_pattern, param).group(1) for param in func_params.split(",")]


                stubs_file.write(f"    def {func_name}({', '.join(func_params)}){func_match.group(3)}:{match.group(3) or ''}\n        ...\n\n")
            elif match.group(4) is not None:
                stubs_file.write(match.group(4))


