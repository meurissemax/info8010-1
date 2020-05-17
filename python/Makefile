# INFO8010-1 - Deep learning
# University of Liège
# Academic year 2019-2020

# Project : neural style transfer

# Authors :
#     - Maxime Meurisse
#     - Adrien Schoffeniels
#     - Valentin Vermeylen

##############
# Techniques #
##############

# Run the Gatys et al. technique
gatys:
	python gatys/main.py

# Run the Cycle GAN technique
cycle-gan:
	python cycle-gan/main.py


###############
# Environment #
###############

# Create the anaconda environment
create-env:
	conda env create -f environment.yml


############
# Cleaning #
############

# Phony
.PHONY: clean

# Delete python residus
clean:
	@find . | grep -E '(__pycache__|\.pyc|\.pyo)' | xargs rm -rf
