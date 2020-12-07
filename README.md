# InnerWave
A project focusing on Inner Wave Detection and Localization

# Data
Remote sensing data from instruments on satellite, containing a number of variables. In total 8 variables including longitude and latitude are selected manually.

## Results

:Model:|:Acc:|:MSE/MAE:
:RF:|:89.4:|:TODO:
:XGBoost:|:91.6:|:3.56/0.72:
:LightGBM:|:90.6:|:3.80/0.77:
:NN:|:90.5:|:TODO:
:ResNet50:|:\:|:TODO:
:VGG16:|:\:|:TODO:

# Methodology
## Classification
Data in 8*32 with 8 variables and a window containing 32 data points. In total 3340 pieces.
Labels are 0/1.

## Localization
Data in 8*16 with 8 variables and a window containing 16 data points. In total 6680 pieces.
Labels are Ints in [0,16] where 0 represents "no Inner Wave" and any other number represents the location of inner wave (which pixel).

Under the error of ±5 pixels, the classification accuracy is 97%.
