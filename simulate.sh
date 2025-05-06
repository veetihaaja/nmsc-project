#!/bin/bash

# this bash script runs the simulation, and then creates a video of the D field

cd output/D_plots
rm *.png
rm *.mp4

cd ../v_plots
rm *.png
rm *.mp4

cd ../..

# run the simulation
./main

# create plots
python3 plotter.py

# create video of D field
cd output/D_plots
ffmpeg -framerate 2 -pattern_type glob -i '*.png'   -c:v libx264 -pix_fmt yuv420p D_field.mp4
mv D_field.mp4 ../../

# create video of v field
cd ../v_plots
ffmpeg -framerate 2 -pattern_type glob -i '*.png'   -c:v libx264 -pix_fmt yuv420p v_field.mp4
mv v_field.mp4 ../../