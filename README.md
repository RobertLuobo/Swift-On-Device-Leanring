# Swift-On-Device-Leanring
A lightweight on-device learning framework for mobile edge devices.

## Install

This project uses [node](http://nodejs.org) and [npm](https://npmjs.com). Go check them out if you don't have them locally installed.

```sh
$ git clone https://github.com/FromSystem/Swift-On-Device-Leanring
```

## Environment

pytorch 		  1.4.0
torchvision 	0.5.0
python		    3.7
numpy
logging			
time
matplotlib
collections

## File description

config.py: All the parameters are set in this file. Please note the logger filepath  and data filepath

quantizer.py: Qcon2d and QLinear

shufflenet.py : model network

Timer.py: logger class to log the information when training

shufflenet_qat.py:  the main file for fake quantizaiton 

## Run



```python
python shufflenet_qat.py
```

