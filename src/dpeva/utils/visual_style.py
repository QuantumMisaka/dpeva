import matplotlib.pyplot as plt
import seaborn as sns

def set_visual_style(font_size: int = 12, context: str = "paper", style: str = "whitegrid"):
    """
    Sets the global visualization style for DPEVA plots.
    
    Args:
        font_size (int): Base font size.
        context (str): Seaborn context ('paper', 'notebook', 'talk', 'poster').
        style (str): Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks').
    """
    # Set Seaborn theme
    sns.set_theme(style=style, context=context)
    
    # Override specific matplotlib params for scientific publication quality
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size,
        'axes.titlesize': font_size + 2,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'lines.linewidth': 1.5,
        # Ensure fonts are editable in PDFs
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
