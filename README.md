# InnerWave
A project focusing on Inner Wave Detection and Localization

# Data
Remote sensing data from instruments on satellite, containing a number of variables. In total 8 variables including longitude and latitude are selected manually.

## Results

|  Model   | Classification (Acc) |  Localization (MSE / MAE)  |
| :------: | :------: | :---------: |
|   :RF:   |   89.4   |    TODO     |
| XGBoost  |   91.6   | 3.56 / 0.72 |
| LightGBM |   90.6   | 3.80 / 0.77 |
|    CNN   |   90.5   |    TODO     |
| ResNet50 |    \     |    TODO     |
|  VGG16   |    \     |    TODO     |





# Methodology

Train models on training set (80%), test on testing set (20%). Run 10 times and get the average.

## Classification
Data in 8*32 with 8 variables and a window containing 32 data points. In total 3340 pieces.
Labels are 0/1.

## Localization
Data in 8*16 with 8 variables and a window containing 16 data points. In total 6680 pieces.
Labels are Ints in [0,16] where 0 represents "no Inner Wave" and any other number represents the location of inner wave (which pixel).
Under the error of ±5 pixels, the highest localization accuracy is 97%.

# Remark
- Obviously, 16-pixels dataset is better than 32-pixels dataset.
- When adjusting the error boundary to ±3 pixels, the localization accuracy is 95%.
- CNN model consists of 5 conv layers (conv+bn+maxpooling) and 3 FC layers, with single node top layer outputing the probability.
