"""
INFO8010-1 - Deep learning
University of Liège
Academic year 2019-2020

Project : neural style transfer

Authors :
    - Maxime Meurisse
    - Adrien Schoffeniels
    - Valentin Vermeylen
"""

###########
# Imports #
###########

import matplotlib.pyplot as plt

from matplotlib import rc


############
# Settings #
############

# LaTeX font style
rc('text', usetex=True)


#############
# Functions #
#############

def line_graph(x, y_list, x_label, y_label, legend, fig_name):
    # Initialize figure
    plt.figure()

    try:
        # Create figure
        fig, ax = plt.subplots()

        # Set the scale
        ax.set_yscale('log')

        # Add the line
        for y in y_list:
            ax.plot(x, y)

        # Add axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Add legend
        ax.legend(legend)

        # Axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')

        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(True, color='#EEEEEE')

        # Tick parameters
        ax.tick_params(bottom=False, left=False)
        ax.set_axisbelow(True)

        # Generate and save figure
        fig.tight_layout()
        fig.savefig(fig_name)
    finally:
        plt.close()
