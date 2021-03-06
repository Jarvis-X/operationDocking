# operationDocking
Final Project for Advanced Aerial Robot, where we show our efforts to drive two quadrotors to follow a trajector, to dock them, and to drive the assembled aerial vehicle to follow a trajectory.

## What do we have
- A powerpoint slide for presentation
- Three sets of simulations with links to the video recordings and simulation plots on https://drive.google.com/drive/folders/1ygl-_h_e2OtyLIPH7fuOZz44HPuPT_Ci?usp=sharing
  - In FlyingOneOmni: trajectory following demonstration using an omnidiretional aerial robot
  - In FlyingTwoQjuad: docking demonstration with two quadrotors following a trajectory
  - In DockingNFlying: locking and flying demonstration from two quadrotors to an omnidirectional robot
- Each subfolder contains one set of simulation, including the CoppeliaSim scene file and the corresponding Python script.
- API library files that are provided by Coppelia Robotics

## What have we done:
- A `Robot` class that supports controlling all kinds of aerial vehicles with at least 4 controllable DoF's
- A min-snap trajectory generator
- A `PID_param` class

## Dependencies
- Python 3.6
- Matplotlib
- numpy
- The inluded CoppeliaSim API files for Python: for more info, visit https://www.coppeliarobotics.com/helpFiles/index.html
