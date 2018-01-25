# Octopus
TuSimple's Autonomous Driving

[![Build Status](https://travis-ci.com/TuSimple/octopus.svg?token=51r12FjC6LsCkZx83e7P&branch=develop)](https://travis-ci.com/TuSimple/octopus)

## Setup Octopus

### 0. Requirement
You must have **Ubuntu 16.04** installed to use Octopus. GPU is needed for gazebo (simulation) mode (otherwise too slow).

### 1. Obtain the source code of Octopus
```shell
git clone git@github.com:TuSimple/octopus.git
```

### 2. Install dependencies
- If you are in TuSimple SD
```
(cd octopus && bash scripts/octopus_install_ubuntu1604.sh)
```
- If you are in TuSimple Beijing

```
(cd octopus && bash scripts/octopus_install_ubuntu1604.sh bj)
```
Then reboot the system.

### 3. Make Octopus
```
(cd octopus/scripts && ./update.sh) && source ~/.bashrc # or ~/.zshrc
```
#### Troubleshooting
##### TS1: If you see error messages related to **flycapture**

```
sudo apt-get install libglademm-2.4-1v5 libgtkmm-2.4-1v5 libgtkmm-2.4-dev libglademm-2.4-dev libgtkglextmm-x11-1.2-dev
sudo apt-get -f install
sudo apt-get remove libflycapture\*
(cd octopus/scripts && ./update.sh) && source ~/.bashrc # or ~/.zshrc
```

If you *still* see error messages related to **flycapture**
```
sudo apt-get -f install
sudo apt-get remove libflycapture\*
(cd octopus/scripts && ./update.sh -4) && source ~/.bashrc # or ~/.zshrc
```

##### TS2: If you also want to use the DL-based perception modules
Make sure you have *CUDA* and *CuDNN* installed
```
(cd octopus/external_packages && ./init.sh)
(cd octopus/scripts && ./update.sh -p) && source ~/.bashrc # or ~/.zshrc
```

### 4. Install Tusimple-Gazebo7
Gazebo7 is installed along with ROS. However, we have modified its source code to fix a critical bug.
```shell
git clone git@github.com:TuSimple/tusimple-gazebo7.git
(cd tusimple-gazebo7 && ./build.sh) # requires ~1 hr
# export path
TUSIMPLE_GAZEBO_INSTALL_PATH=$(pwd)/tusimple-gazebo7/install
echo "export PATH=$TUSIMPLE_GAZEBO_INSTALL_PATH/bin:"'$PATH' >> ~/.bashrc # or ~/.zshrc
echo "export LD_LIBRARY_PATH=$TUSIMPLE_GAZEBO_INSTALL_PATH/lib:"'$LD_LIBRARY_PATH' >> ~/.bashrc # or ~/.zshrc
echo "export PKG_CONFIG_PATH=$TUSIMPLE_GAZEBO_INSTALL_PATH/lib/pkgconfig:"'$PKG_CONFIG_PATH' >> ~/.bashrc # or ~/.zshrc
source ~/.bashrc # or ~/.zshrc
```
Run `gazebo` to see if gazebo can be launched, then close it.
#### 4.1 Install Joystick Module
##### Equipment List:
1. Playstation DualShock 4 Remote Controller (other type of remote controller might work too)
2. Usb to Micro-usb cable
##### Install Package
```shell
sudo apt-get install ros-kinetic-joy
```
##### Configuring the Joystick
Connect your joystick to your computer. Now let's see if Linux recognized your joystick.
```shell
ls /dev/input/
```
You will see a listing of all of your input devices similar to below:
```
by-id    event0  event2  event4  event6  event8  mouse0  mouse2  uinput
by-path  event1  event3  event5  event7  js0     mice    mouse1
```
As you can see above, the joystick devices are referred to by jsX ; in this case, our joystick is js0.
Now let's make the joystick accessible for the ROS joy node.
```shell
ls -l /dev/input/js0
sudo chmod a+rw /dev/input/js0
```
Starting the Joy Node
```shell
roscore
rosparam set joy_node/dev "/dev/input/js0"
rosrun joy joy_node
```
In new terminal, try to echo the joy topic
```shell
rostopic echo joy
```
As you move the joystick around, if you can see the topic is updating, then you're all set

### 5. Have fun!
```shell
roslaunch console_backend console.launch
```
Open http://localhost:8080/#/, switch to the *console* tab and enjoy!
