import sys
import time
from dpeva import __version__

# ANSI Color Codes (Based on Brand Specs: Blue, Orange, Neon Green)
BLUE = "\033[38;2;110;107;255m"   # DeepModeling Blue
ORANGE = "\033[38;2;255;138;0m"   # Warning Orange
GREEN = "\033[38;2;57;255;20m"    # Neon Green
GRAY = "\033[90m"
RESET = "\033[0m"

r'''
Banner Reference:

┌────────────────────────────────────────────────────────────────────────────┐
│  ____  ____       _______     __    _       [ STATUS: ACTIVE ]             │
│ |  _ \|  _ \     | ____\ \   / /   / \      ⚡ SYNCHRO: 100%               │
│ | | | | |_) |____|  _|  \ \ / /   / _ \     ◌ SAMPLE: SAMPLING...          │
│ | |_| |  __/_____| |___  \ V /   / ___ \    >> AWAKENING CHECK <<          │
│ |____/|_|        |_____|  \_/   /_/   \_\   v0.6.5-Execution-Ready         │
├────────────────────────────────────────────────────────────────────────────┤
│  :: LCL Sea Density ::  ......  * .  ..  .  *  .  * .  ...  * .  ..  .     │
│  Deep Potential EVolution Accelerator          >>> [PRESS START] <<<       │
└────────────────────────────────────────────────────────────────────────────┘

Banner Basic Reference:

  ____  ____       _______     __    _   
 |  _ \|  _ \     | ____\ \   / /   / \  
 | | | | |_) |____|  _|  \ \ / /   / _ \ 
 | |_| |  __/_____| |___  \ V /   / ___ \
 |____/|_|        |_____|  \_/   /_/   \_\

'''


def show_banner(no_delay=False):
    """
    Display the ASCII Art Banner for DP-EVA.

    Args:
        no_delay (bool): If True, skips the sleep delay after printing.
    """
    # Version string formatting
    ver_text = f"v{__version__}-Execution-Ready"
    # Ensure it fits within 26 chars (leaving 2 chars for gap)
    if len(ver_text) > 26:
        ver_text = ver_text[:26]

    # --- Precise Alignment Configuration ---
    # Total Inner Width: 76 chars (Border is 78 wide: 1 + 76 + 1)
    # ASCII Art Width: 42 chars (Fixed)
    # Gap: 2 chars
    # Status Width: 32 chars (76 - 42 - 2)
    
    W_AA = 42
    W_ST = 32
    
    # Helper to pad status text correctly (handling emoji width)
    def pad_status(text, width=W_ST):
        # Calculate visual length (simplified for known chars)
        visual_len = len(text)
        if "⚡" in text: visual_len += 1  # Emoji usually takes 2 spaces visually but len is 1
        # if "◌" in text: visual_len += 1   # Dotted Circle is usually 1 char wide in monospace
        
        padding = width - visual_len
        if padding < 0: padding = 0
        return text + " " * padding

    # Row 1
    # AA: 42 chars
    # Split manually to color: "  ____  ____       " (Blue) + "_______     __    _    " (Orange)
    aa1_left = "  ____  ____       "
    aa1_right = "_______     __    _    "
    st1_txt = "[ STATUS: ACTIVE ]"
    
    aa1 = f"{BLUE}{aa1_left}{RESET}{ORANGE}{aa1_right}{RESET}"
    st1 = f"{GRAY} {pad_status(st1_txt, W_ST-1)}{RESET}" # -1 for leading space
    
    # Row 2
    # AA: 42 chars
    # Split manually to color: " |  _ \|  _ \     " (Blue) + "| ____\ \   / /   / \   " (Orange)
    aa2_left = r" |  _ \|  _ \     "
    aa2_right = r"| ____\ \   / /   / \   "
    st2_txt = "⚡ SYNCHRO: 100%"
    
    aa2 = f"{BLUE}{aa2_left}{RESET}{ORANGE}{aa2_right}{RESET}"
    st2 = f"{GRAY} {pad_status(st2_txt, W_ST-1)}{RESET}"

    # Row 3
    # AA: 42 chars
    # aa3_txt = " | | | | |_) |____|  _|  \ \ / /   / _ \  "
    st3_txt = "◌ SAMPLE: SAMPLING..."
    
    aa3 = fr"{BLUE} | | | | |_) |{RESET}{ORANGE}____{RESET}{ORANGE}|  _|  \ \ / /   / _ \  {RESET}"
    st3 = f"{GRAY} {pad_status(st3_txt, W_ST-1)}{RESET}"

    # Row 4
    # AA: 42 chars
    # aa4_txt = " | |_| |  __/_____| |___  \ V /   / ___ \ "
    st4_txt = ">> AWAKENING CHECK <<"
    
    aa4 = fr"{BLUE} | |_| |  __/{RESET}{ORANGE}_____{RESET}{ORANGE}| |___  \ V /   / ___ \ {RESET}"
    st4 = f"{GREEN} {pad_status(st4_txt, W_ST-1)}{RESET}"

    # Row 5
    # AA: 41 chars (Needs 1 space pad)
    # aa5_txt = " |____/|_|        |_____|  \_/   /_/   \_" 
    st5_txt = ver_text
    
    aa5 = fr"{BLUE} |____/|_|        {RESET}{ORANGE}|_____|  \_/   /_/   \_\{RESET}" # Corrected length to 42
    st5 = f"{GRAY} {pad_status(st5_txt, W_ST-1)}{RESET}"

    banner = f"""
{GRAY}┌────────────────────────────────────────────────────────────────────────────┐{RESET}
{GRAY}│{RESET}{aa1}  {st1}{GRAY}│{RESET}
{GRAY}│{RESET}{aa2}  {st2}{GRAY}│{RESET}
{GRAY}│{RESET}{aa3}  {st3}{GRAY}│{RESET}
{GRAY}│{RESET}{aa4}  {st4}{GRAY}│{RESET}
{GRAY}│{RESET}{aa5}  {st5}{GRAY}│{RESET}
{GRAY}├────────────────────────────────────────────────────────────────────────────┤{RESET}
{GRAY}│{RESET}  {GRAY}:: LCL Sea Density ::{RESET}  {ORANGE}......{RESET}  {BLUE}*{RESET} {ORANGE}.{RESET}  {ORANGE}..{RESET}  {ORANGE}.{RESET}  {BLUE}*{RESET}  {ORANGE}.{RESET}  {BLUE}*{RESET} {ORANGE}.{RESET}  {ORANGE}...{RESET}  {BLUE}*{RESET} {ORANGE}.{RESET}  {ORANGE}..{RESET}  {ORANGE}.{RESET}     {GRAY}│{RESET}
{GRAY}│{RESET}  {GRAY}Deep Potential EVolution Accelerator{RESET}          {ORANGE}>>> [PRESS START] <<<{RESET}       {GRAY}│{RESET}
{GRAY}└────────────────────────────────────────────────────────────────────────────┘{RESET}
"""
    print(banner)

    if not no_delay:
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    show_banner(no_delay=True)
