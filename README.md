# CapsATTEND-GAN
### Data
We pretrain our models using [Microsoft COCO Dataset](http://cocodataset.org/#download). 
Then, we train the models using [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/).

### Requirements
1. Python 3
2. Numpy
3. Hickle
4. Python-skimage
3. Tensorflow 1.14

### TODO
- [x] Build LSTM Generator
- [x] Incorporate emotions into the Generator
- [x] Generator Logger
- [x] Build Conventional Discriminator
- [x] Discriminator Logger
- [ ] GAN train engine
- [ ] Build CapsNet Discriminator
- [ ] ...

### Train
1. Download [Microsoft COCO Dataset](http://cocodataset.org/#download) including neutral image caption data: images: 2014 Train images [83K/13GB] ([download](http://images.cocodataset.org/zips/train2014.zip)), 2014 Val images [41K/6GB] ([download](http://images.cocodataset.org/zips/val2014.zip)), 2014 Test images [41K/6GB] ([download](http://images.cocodataset.org/zips/test2014.zip)), captions: 2014 Train/Val annotations [241MB] ([download](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)) and extract them to the folder data/images.
2. Download [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/) including sentiment-bearing image caption data: captions ([download](http://users.cecs.anu.edu.au/~u4534172/data/Senticap/senticap_dataset.zip)) and only extract the file senticap_dataset.json to data/annotations.
3. Run `python resize.py` (reseizes the downloded images into [224, 224] and puts them in data/images).
4. Run `python train.py` to train the generator and discriminator.

### Test

### Results
