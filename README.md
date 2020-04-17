# CapsATTEND-GAN
### Data
We pretrain our models using [Microsoft COCO Dataset](http://cocodataset.org/#download). 
Then, we train the models using [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/).

### Requirements
1. python 3.7.4
2. numpy 1.18.1
3. hickle 3.4.6
4. scikit-image 0.16.2
5. tensorflow 1.14 or tensorflow-gpu 1.14
6. tqdm 4.44.1
7. torch 1.4.0
8. matplotlib 3.1.3

### TODO
- [x] COCO Dataset loader and build pre-processing engine
- [x] Build LSTM Generator
- [x] Incorporate emotions into the Generator
- [x] Generator Logger
- [x] Build Conventional Discriminator
- [x] Discriminator Logger
- [x] GAN train engine
- [x] Validation engines
- [ ] Record examples of generated captions in GAN structure
- [ ] SentiCap Dataset loader and build pre-processing engine
- [ ] Build CapsNet Discriminator
- [ ] Inference engine
- [ ] Plots

### Train
1. Download [Microsoft COCO Dataset](http://cocodataset.org/#download) including neutral image caption data: images: 2014 Train images [83K/13GB] ([download](http://images.cocodataset.org/zips/train2014.zip)), 2014 Val images [41K/6GB] ([download](http://images.cocodataset.org/zips/val2014.zip)), 2014 Test images [41K/6GB] ([download](http://images.cocodataset.org/zips/test2014.zip)), captions: 2014 Train/Val annotations [241MB] ([download](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)) and extract them to the folder data/images.
2. Download [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/) including sentiment-bearing image caption data: captions ([download](http://users.cecs.anu.edu.au/~u4534172/data/Senticap/senticap_dataset.zip)) and only extract the file senticap_dataset.json to data/annotations.
3. Run `python resize.py` (reseizes the downloded images into [224, 224] and puts them in data/images).
4. Run `python train.py` to train the generator and discriminator.

### Test

### Results
