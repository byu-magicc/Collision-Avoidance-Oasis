# Collision Avoidance

## Description
A repo exploring various algorithms to avoid collision with moving obstacles.

## Installation
No special installation instructions yet.

## Authors and acknowledgment
James Adams, Brad Nelson, JJ Liu, Randy Beard

## Project status
Development Paused



This repo explores many different estimators designed to estimate the position and velocity of constant velocity intruders. 

The root directory holds three different approaches. 

The first is a particle filter approach that utilizes bearing and time-to-collision to estimate the family of intruders and avoid the entire family. This is found in [particle_filter_improved.py](other/particle_filter_improved.py). The mathematical details of the particle filter algorithm can be found in Chapter 4 of the thesis of James Adams titled *A Series of Improved and Novel Methods in Computer Vision Estimation* (link coming soon).

The second is an intruder wedge area estimation, found in the "Wedge Avoidance" folder. This approach uses time-to-collision and the camera information (bearing, pixel size) to bound the range of an intruder and estimate the intruders future trajectory, using 4-sided wedges to block out the airspace, allowing a path planner to safely avoid the intruder.

The third is an optimization approach which minimizes the probability of collision when given intruder trajectories. This is found in the folder "Trajectory Optimize"