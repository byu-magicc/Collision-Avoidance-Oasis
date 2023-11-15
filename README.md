# Collision Avoidance

## Description
A repo exploring various algorithms to avoid collision with moving obstacles.

## Installation
No special installation instructions yet.

## Authors and acknowledgment
James Adams

## Project status
Actively developing.

This repo explores many different estimators designed to estimate the position and velocity of constant velocity intruders. They are found in the root directory. We also implemented a particle filter approach that utilizes bearing and time-to-collision to estimate the family of intruders and avoid the entire family. This is found in [particle_filter_improved.py](other/particle_filter_improved.py). The mathematical details of the particle filter algorithm can be found in Chapter 4 of the thesis of James Adams titled *A Series of Improved and Novel Methods in Computer Vision Estimation* (link coming soon).