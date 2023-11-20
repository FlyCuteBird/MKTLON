# Introduction
Source code of the Multi-level knowledge-driven feature representation and triplet loss optimization network (the text embedding is BERT).
## Requirements and Installation
The following dependencies are recommended.

* Python==3.7.0
* pytorch==1.7.0
* torchvision==0.8.0
* torchaudio==0.7.0
* pytorch-pretrained-bert==0.6.2
  
## Pretrained model
If you don't want to train from scratch, you can download the pre-trained MKTLON model  from [here]([https://drive.google.com/drive/folders/1eddbVAGbjHvofX96FuY4Sq7YcJTmuMhV?usp=drive_link](https://drive.google.com/drive/folders/1_ajDOu57KVDaGqSDia0jO-kcr4p1l1No?usp=sharing])(for Flickr30K)
```bash
i2t: 503.0
Image to text: 76.7  94.2  97.6
Text to image: 59.0  84.8  90.7
t2i: 515.2
Image to text: 79.9  95.6  97.8
Text to image: 62.5  87.2  92.2
```
## Download Data 
We utilize the image feature created by SCAN, downloaded [here](https://github.com/kuanghuei/SCAN). Some related text data can be found in the 'data' folder of the project (for Flickr30K).

## Training 
```bash
python train.py 
```
## Testing
```bash
python test.py
```
