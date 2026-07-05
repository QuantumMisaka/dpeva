import re

from dpeva.utils.banner import show_banner


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
DOUBLE_WIDTH_CHARS = {"\u26a1"}


def _strip_ansi(text):
    return ANSI_ESCAPE_RE.sub("", text)


def _visual_width(text):
    return sum(2 if char in DOUBLE_WIDTH_CHARS else 1 for char in text)


def test_banner_mentions_project_author(capsys):
    show_banner(no_delay=True)

    plain_banner = _strip_ansi(capsys.readouterr().out)

    assert "@QuantumMisaka" in plain_banner


def test_banner_preserves_fixed_visual_width(capsys):
    show_banner(no_delay=True)

    plain_banner = _strip_ansi(capsys.readouterr().out)
    banner_lines = [line for line in plain_banner.splitlines() if line]

    assert banner_lines
    assert all(_visual_width(line) == 78 for line in banner_lines)
