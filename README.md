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
- [x] Record examples of generated captions in GAN structure
- [x] SentiCap Dataset loader and build pre-processing engine
- [ ] Build CapsNet Discriminator
- [ ] Inference engine
- [ ] Plots

### Train
1. Run `./download.sh` and go to step 4, otherweise go to step 5.
2. Download [Microsoft COCO Dataset](http://cocodataset.org/#download) including neutral image caption data: images: 2014 Train images [83K/13GB] ([download](http://images.cocodataset.org/zips/train2014.zip)), 2014 Val images [41K/6GB] ([download](http://images.cocodataset.org/zips/val2014.zip)), 2014 Test images [41K/6GB] ([download](http://images.cocodataset.org/zips/test2014.zip)), captions: 2014 Train/Val annotations [241MB] ([download](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)) and extract them to the folder data/images.
3. Download [SentiCap Dataset](http://cm.cecs.anu.edu.au/post/senticap/) including sentiment-bearing image caption data: captions ([download](http://users.cecs.anu.edu.au/~u4534172/data/Senticap/senticap_dataset.zip)) and only extract the file data/senticap_dataset.json to data/annotations.
4. Download the VGG network used for feature extraction [download](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) and move it to the folder data/
5. Run `python resize.py --input_folder_dir ./data/images/train2014/ --output_folder_dir ./data/images/train2014_resized/ && python resize.py --input_folder_dir ./data/images/val2014/ --output_folder_dir ./data/images/val2014_resized/` (reseizes the downloded images into [224, 224] and puts them in data/images).
6. Run `python prepro.py --coco_dataset_portions 1. 0.8 0.2 --senticap_dataset_portions 0.8 0.19 0.01`, where the first second and third entries are the split portion from the original dataset.
7. Run `python train.py --gen_train --gen_save_model_dir ./model/generator/ --gen_dataset coco --gen_batchsize 8 --gen_epochs 10` to pretrain the generator.
8. Run `python train.py --disc_train --disc_save_model_dir ./model/discriminator/ --disc_dataset coco --disc_batchsize 8 --disc_epochs 10` to pretrain the discriminator.
9. Run `python train.py --gan_train --gan_save_model_dir ./model/gan/ --gan_dataset senticap --gan_batchsize 8 --gan_epochs 10` to train the GAN. You can add the arguments `--gen_load_model_dir` and/or `--disc_load_model_dir` to initialize your model with a pretrained generator and/or discriminator.

### Test

### Results
