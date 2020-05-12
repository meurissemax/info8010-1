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

from graph import line_graph

from gatys.gatys import gatys


#####################
# General variables #
#####################

# Gatys et al.
style_path = '../resources/images/style/picasso.png'
content_path = '../resources/images/content/eiffel-tower.png'
output_path = 'outputs/gatys/picasso-eiffel.png'


########
# Main #
########

if __name__ == '__main__':
    # Print general information
    print('-------------------------------------------------')
    print('| Deep learning project - Neural style transfer |')
    print('-------------------------------------------------')
    print()

    # Run Gatys et al. algorithm
    gatys(style_path, content_path, output_path, line_graph)
