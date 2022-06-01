# Internal Solitary Wave Localization

## Code Usage
Run experiments using codes below:
```
python data_process.py
python training_testing.py
```

Run pretraining experiments using codes below:
```
python data_process.py
python pretraining.py
python training_testing.py
```


There are demo codes in above scripts on how to conduct experiments with different settings. You can modify specific parameters such as oversampling techniques, learning rate, model type, pretraining model, etc. Parameters are as following:


data_process.py
- data_shape: "1x96"/"16x16x1"
- oversampling: ""/"oversample"/"SMOTE"/"BorderlineSMOTE"/"ADASYN"


training_testing.py:
- model_name: "RF"/"LGB"/"MLP"/"ResNet"/"BoTNet"
- data_shape: "1x96"/"16x16x1"
- oversampling: ""/"oversample"/"SMOTE"/"BorderlineSMOTE"/"ADASYN"
- loss_func: "CE"/"weightedLoss"/"FocalLoss"
- lr
- epoch
- batch_size

## Reference
- Pretraining codes utilises an open-sourced SimCLR library: 
  
  https://github.com/Spijkervet/SimCLR
- ResNet codes utilises following implementations: 

  https://github.com/leaderj1001/BottleneckTransformers/blob/main/model.py
  
  https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
