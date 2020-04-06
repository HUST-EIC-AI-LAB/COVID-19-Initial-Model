# COVID-19
This repo is for the experiment codes.     

We regard COVID-19 diagnosis as a multi-classification task. Two models are builted for this task, the first one can classify COVID-19 to four classes including `healthy`, `COVID-19`, `other viral pneumonia`, `bacterial pneumonia`, and the other one can accurately assess the severity level for putative COVID-19 patients.   

As for our datasets, we use multi-center datasets of totally 1276 CT studies comprising 432 confirmed COVID-19, 76 other viral pneumonia, 350 bacterial pneumonia and 418 healthy cases. 



### Data Preprocess
Codes for data preprocess is in:  `./utils`    

Initialy, the CT images we get from hospital are not exactly what we can feed to network directly. So there are some cleaning operations have been done to the initial datasets. These operations are built based on our careful check into the CT images. We found when the slice numbers of CT are less than 15 and the width or height of CT images are not equal to 512 pixeles, the images are usually not useful lung CT. After that we clip all images' pixels to [-1200, 600], which is a ordinay operation in medical image. Finally, we calculate the mean and std of the whole datasets and then normalize every image.    


### Model  
We ultilize 3D-DenseNet as our baseline model. Before we feed images into network, we find if we can cover the whole lung in temporal direction, the model behave much better. Besides, we confim that there is a linear relation between slice thickness and slice numbers. As a result, a sample strategy is bulit which follows the following pseudo codes:  
```python
if slice number<=80:
    random start index;
    choose image every 1 interval; # if start=0, choose [0,1,2,...,13,14,15]
elif slice number<=160:
    random start index;
    choose image every 5 interval; # if start=0, choose [0,4,9,...,64,69,74]
else:
    start=random.randrange(0,z_train-160)
    random start index;
    choose image every 10 interval; # if start=0, choose [0,9,19,...,129,139,149] 
```
- resize sequence images to [16,128,128]
- no augmentation
- regulization --- linear_scheduler dropblock(prob=0., block_size=5 )
- Optimizer --- torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-5)
- lr_scheduler --- lr_warmup and optim.lr_scheduler.CosineAnnealingLR()
- output layer --- FC(features, 4) -> weighted cross entropy + OHEM?
- epoch --- 100
- batch size --- 64
- Machine Resource --- 2 Tesla V100 6 hours  


### Training  
To train the model(s) in the paper, run this command:
```sh
python train.py
```


### Test
To evaluate my model on our dataset, run this command:
```sh
python test_case.py
```




### Results
Our model achieves the following performance on :

- four classes  


- six classes


- test on XieHe datasets









