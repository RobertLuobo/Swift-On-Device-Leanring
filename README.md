# Swift-On-Device-Leanring
A lightweight on-device learning framework for mobile edge devices.

## Install

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

### python file

config.py: All the parameters are set in this file. Please note the logger filepath  and data filepath

quantizer.py: Qcon2d and QLinear

shufflenet.py : model network

Timer.py: logger class to log the information when training

shufflenet_qat.py:  the main file for fake quantizaiton 

###  file

data: datasets are stored in this file

./ShuffleNet_QAT/logger: this is a place where you can find out the running infomation.

## Run



```python
python shufflenet_qat.py
```

