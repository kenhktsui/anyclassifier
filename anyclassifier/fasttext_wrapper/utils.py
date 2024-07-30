import re


def replace_newlines(text: str) -> str:
    return re.sub("\n+", " ", text)
