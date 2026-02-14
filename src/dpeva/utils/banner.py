import sys
import time
import shutil
from dpeva import __version__

# ANSI Colors
GREEN = "\033[1;32m"
CYAN = "\033[1;36m"
WHITE = "\033[1;37m"
RESET = "\033[0m"

# Project Info
PROJECT_URL = "https://github.com/QuantumMisaka/dpeva"
BANNER_WIDTH = 74

def get_terminal_size():
    return shutil.get_terminal_size((80, 24))

def center_text(text, width):
    if len(text) >= width:
        return text
    pad = (width - len(text)) // 2
    return " " * pad + text + " " * (width - len(text) - pad)

def show_banner(no_delay=False):
    """
    Displays the ASCII Art Banner for DP-EVA.
    """
    term_w, term_h = get_terminal_size()
    
    # Ensure banner fits
    if term_w < BANNER_WIDTH or term_h < 15:
        return

    # ASCII Art Title
    title_art = [
        r"  ____  ____       _______     __    _    ",
        r" |  _ \|  _ \     | ____\ \   / /   / \   ",
        r" | | | | |_) |____|  _|  \ \ / /   / _ \  ",
        r" | |_| |  __/_____| |___  \ V /   / ___ \ ",
        r" |____/|_|        |_____|  \_/   /_/   \_" + "\\",
    ]

    # Subtitle
    subtitle = "Deep Potential EVolution Accelerator"

    # Border Construction
    border_top = "+" + "-" * (BANNER_WIDTH - 2) + "+"
    border_bottom = "+" + "-" * (BANNER_WIDTH - 2) + "+"
    
    # Build Lines
    lines = []
    lines.append(WHITE + border_top + RESET)
    lines.append(WHITE + "|" + " " * (BANNER_WIDTH - 2) + "|" + RESET)
    
    # Title
    for line in title_art:
        centered_line = center_text(line, BANNER_WIDTH - 2)
        lines.append(WHITE + "|" + GREEN + centered_line + WHITE + "|" + RESET)
        
    lines.append(WHITE + "|" + " " * (BANNER_WIDTH - 2) + "|" + RESET)
    
    # Subtitle
    centered_sub = center_text(subtitle, BANNER_WIDTH - 2)
    lines.append(WHITE + "|" + CYAN + centered_sub + WHITE + "|" + RESET)
    
    lines.append(WHITE + "|" + " " * (BANNER_WIDTH - 2) + "|" + RESET)
    
    # Version & Link
    ver_str = f"Version {__version__}"
    link_str = PROJECT_URL
    
    lines.append(WHITE + "|" + center_text(ver_str, BANNER_WIDTH - 2) + "|" + RESET)
    lines.append(WHITE + "|" + center_text(link_str, BANNER_WIDTH - 2) + "|" + RESET)
    lines.append(WHITE + "|" + " " * (BANNER_WIDTH - 2) + "|" + RESET)
    lines.append(WHITE + border_bottom + RESET)

    # Print Banner
    print("\n" * 1)
    for line in lines:
        print(line)
    print("\n")

    if not no_delay:
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    show_banner(no_delay=True)
