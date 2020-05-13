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

# Algorithms
from gatys.gatys import Gatys

# Function to make plots
from graph import line_graph as graph


#####################
# General variables #
#####################

# Gatys et al.
style = '../resources/images/style/starry-night.png'
content = '../resources/images/content/eiffel-tower.png'
output = 'outputs/gatys/starry-night-eiffel-tower.png'


########
# Main #
########

if __name__ == '__main__':
    #######################
    # General information #
    #######################

    print('#################################################')
    print('# Deep learning project - Neural style transfer #')
    print('#################################################')
    print()

    ##########################
    # Gatys et al. algorithm #
    ##########################

    g = Gatys()

    g.initialize()

    g.run(style, content)
    g.export(output, graph)
