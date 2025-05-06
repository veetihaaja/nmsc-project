#!/bin/bash

# this bash script runs the simulation, and then creates a video of the D field

rm *.mp4

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
cat *.png | ffmpeg -f image2pipe -framerate 3 -i - D_field.mp4
mv D_field.mp4 ../../

# create video of v field
cd ../v_plots
cat *.png | ffmpeg -f image2pipe -framerate 3 -i - v_field.mp4
mv v_field.mp4 ../../