import os
from utils_constants import IMAGES_PATH, FIG_SIZE, C2
import matplotlib.pyplot as plt
from utils_colors import Colors

def save_plot(name, ext="png", res=300):
    path = os.path.join(IMAGES_PATH, name + "." + ext)
    plt.tight_layout()
    plt.savefig(path, format=ext, dpi=res)
    print(f"{Colors.GREEN}Image {Colors.RES}{name} {Colors.GREEN}saved{Colors.RES}")

def save_and_show(name):
    save_plot(name)
    plt.show()

def scatter_plot(data, name):
        data.plot(kind='scatter', x='km', y='price', alpha=0.5, color='green', figsize=FIG_SIZE)
        save_and_show(name)

def histogram_plot(data, name):
        data.hist(bins=6, figsize=FIG_SIZE, color=C2) # 6 car 24 paires de données, donc pratique
        save_and_show(name)
