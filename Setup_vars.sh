#!/bin/bash -l

source /work/ta116/shared/users/tetts_ta/python3.9/bin/activate #sets up python environment
module load cdo # cdo tool - good for some data processing but hard to use
module load ncview # ncview - handy for quick looks at data
module load imagemagick # provides display - good for viewing images

echo "This ran - Yay! But don't forget to activate the python environment (source /work/ta116/shared/users/tetts_ta/python3.9/bin/activate)... Good luck! "
