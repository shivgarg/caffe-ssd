# OpenImages Visual Relationship Challenge

This repo is contains the caffe code for [Single Shot Detector](https://github.com/weiliu89/caffe). The branches contain my experiments with the code base. Original code is present in [ssd](https://github.com/shivgarg/caffe-ssd/tree/ssd) branch.  
This branch (openimages) has been used to train models for [Google AI Open Images - Visual Relationship Track](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track) competition held as a [workshop](https://storage.googleapis.com/openimages/web/challenge.html) in [ECCV, 2018](https://eccv2018.org/).The parent [repo](https://github.com/shivgarg/OpenImages) contains all of the dataset processing code.  
The solution consists of two modules :-
1.  Handles the "is" relation prediction. The model used is a simple VGG16-SSD.
2.  Handles the rest of relations. This module is a two stage network with first stage producing region proposals and the second stage detecting objects in the region proposals. For simplicity, one may consider it to be similar to RCNN object detection architecture though there are major differences between these two approaches. I will add detailed figures/details later. Feel free to raise an issue if you need the information urgently. 

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/shivgarg/caffe-ssd.git
  cd caffe-ssd
  git checkout openimages
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
  ```

### Preparation
1. It is assumed that the dataset has been converted to VOC format and dataset trainval and test split for the modules(is, region->crop) to be trained has been generated. Please follow the guide [here](https://github.com/shivgarg/OpenImages).

3. Create the LMDB file.
  ```Shell
  cd $CAFFE_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/OpenImages/OpenImages_<module>/
  ./data/OpenImages/OpenImages_<module>/create_list.sh  <path to OI in VOC Format>
  
  # It will create lmdb files for trainval and test with encoded original image:
  #   - examples/OpenImages/OpenImages_<module>/OpenImages_<module>_trainval_lmdb
  ./data/OpenImages/OPenImages_<module>/create_data.sh <data_root_dir> <root_dir>
  For root_dir, use path where data related files will be stored. Use ./ if unsure about this. 
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/OpenImages/OpenImages_<module>/SSD_300x300
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/OpenImages/OpenImages_<module>/SSD_300x300/
  python examples/OpenImages_3way/ssd_openimages_<module>.py
  ```
  
2. Infer using the most recent snapshot.
  ```Shell
  # If you would like to generate kaggle compatible output file:
  # Module 1
  python examples/OpenImages_3way/ssd_detect_is.py --gpu_id <id> --labelmap_file <path> --model_def <path to prototxt> --model_weights <path to caffemodel> --image_dir <dir of test images> --output_file <filename>
  
  #Module 2
  python examples/OpenImages_3way/ssd_detect_crop.py --gpu_id <id> --labelmap_region <path> --labelmap_crop <path> --proto_region <path to rpn prototxt> --proto_crop <path to object detector prototxt> --model_region <path to caffemodel> --model_crop <path to caffemodel> --image_dir <dir containing images> --output_file <filename> --relationship_file <candidate relations>
  
  # For combining the output of the both the modules, use the following:
  python combine.py <module 1 output> <moudle 2 output> <image dir> <output file name>
  ```