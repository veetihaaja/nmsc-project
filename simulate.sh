#! /bin/bash

#this bash script compiles the program, runs it and creates two plot videos.
make
./main
python3 Vplotter.py
python3 Dplotter.py