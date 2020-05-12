# INFO8010-1 - Deep learning
# University of Liège
# Academic year 2019-2020

# Project : neural style transfer

# Authors :
#     - Maxime Meurisse
#     - Adrien Schoffeniels
#     - Valentin Vermeylen

# Run the main script
main:
	python main.py

# Create the anaconda environment
env: environment.yml
	conda env create -f environment.yml

# Phony
.PHONY: clean

# Clean the workspace
clean:
	@rm -f -r __pycache__/
	@rm -f -r **/__pycache__/
	@rm -f -r **/**/__pycache__/
