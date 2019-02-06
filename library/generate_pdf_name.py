import re

title = """FAdaptive Tile Coding for Value Function Approximation"""
first_author = """Shimon Whiteson"""

title = "_".join(title.strip().lower().split())
first_author = re.sub(r"(?:\s*)(\w)(?:[\w\W]*)\s(\S+)", r"\1_\2", first_author.strip().lower())

print(f"{first_author}_{title}")
