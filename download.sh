wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip -P data/
unzip data/captions_train-val2014.zip -d data/
rm data/captions_train-val2014.zip

wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip -P data/
unzip data/train2014.zip -d data/
rm data/train2014.zip 

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P data/
unzip image/val2014.zip -d data/
rm data/val2014.zip

wget http://users.cecs.anu.edu.au/~u4534172/data/Senticap/senticap_dataset.zip -P data/
unzip data/senticap_dataset.zip -d data/	
mv data/senticap_dataset/data/senticap_dataset.json data/annotations/