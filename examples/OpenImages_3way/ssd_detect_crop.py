#encoding=utf8
'''
Detection with SSD
In this example, we will load a SSD model and use it to detect objects.
'''

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw
# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

class CaffeDetection:
    def __init__(self, gpu_id, model_def, model_weights, image_resize, labelmap_file):
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

        self.image_resize = image_resize
        # Load the net in the test phase for inference, and configure input preprocessing.
        self.net = caffe.Net(model_def,      # defines the structure of the model
                             model_weights,  # contains the trained weights
                             caffe.TEST)     # use test mode (e.g., don't perform dropout)
         # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104, 117, 123])) # mean pixel
        # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_raw_scale('data', 255)
        # the reference model has channels in BGR order instead of RGB
        self.transformer.set_channel_swap('data', (2, 1, 0))

        # load PASCAL VOC labels
        file = open(labelmap_file, 'r')
        self.labelmap = caffe_pb2.LabelMap()
        text_format.Merge(str(file.read()), self.labelmap)

    def detect(self, image_file, conf_thresh=0.5, topn=5):
        '''
        SSD detection
        '''
        # set net to batch size of 1
        # image_resize = 300
        if type(image_file) == str:
            image = caffe.io.load_image(image_file)
        else:
            image = image_file
        #Run the net and examine the top_k results
        self.net.blobs['data'].reshape(1, 3, self.image_resize, self.image_resize)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        detections = self.net.forward()['detection_out']

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.6.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(self.labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]
        top_ymin = det_ymin[top_indices]
        top_xmax = det_xmax[top_indices]
        top_ymax = det_ymax[top_indices]

        result = []
        for i in xrange(min(topn, top_conf.shape[0])):
            xmin = top_xmin[i] # xmin = int(round(top_xmin[i] * image.shape[1]))
            ymin = top_ymin[i] # ymin = int(round(top_ymin[i] * image.shape[0]))
            xmax = top_xmax[i] # xmax = int(round(top_xmax[i] * image.shape[1]))
            ymax = top_ymax[i] # ymax = int(round(top_ymax[i] * image.shape[0]))
            score = top_conf[i]
            label = int(top_label_indices[i])
            label_name = top_labels[i]
            result.append([xmin, ymin, xmax, ymax, label, score, label_name])
        return result

def main(args):
    '''main '''
    region = CaffeDetection(args.gpu_id,
                               args.proto_region, args.model_region,
                               args.image_resize, args.labelmap_region)
    crop = CaffeDetection(args.gpu_id, args.proto_crop,
                            args.model_crop,args.image_resize,args.labelmap_crop)
    
    relationship_triplet = open(args.relationship_file)
    relationship_triplet = [line.strip('\n') for line in relationship_triplet]
    print relationship_triplet
    output = open(args.output_file,'w')
    output.write('ImageID,Label1,Label2,Relationship,Confidence,xmin1,xmax1,ymin1,ymax1,xmin2,xmax2,ymin2,ymax2\n')
    for _,_,files in os.walk(args.image_dir):
        for f in files:
            result = region.detect(args.image_dir+'/'+f,conf_thresh=0.2)
            print "region ", len(result), f
            img = Image.open(args.image_dir+'/'+f)
            width, height = img.size
            
            for item in result:
                xmin = int(round(item[0] * width))
                ymin = int(round(item[1] * height))
                xmax = int(round(item[2] * width))
                ymax = int(round(item[3] * height))
                
                img.crop((xmin,ymin,xmax,ymax)).save('tmp.jpg')
                result1 = crop.detect('tmp.jpg',topn=20,conf_thresh=0)
                print "crop ",len(result1)
                width_crop = xmax-xmin
                height_crop = ymax-ymin
                score = item[5]
                if len(result1) < 2:
                    continue
                pairs = []
                i = 0
                for it in result1:
                    pairs.append([i,it[6],it[5]])
                    i+=1
                triples = []
                num_pairs = len(pairs)
                for i in xrange(num_pairs):
                    for j in xrange(i+1,num_pairs):
                        if pairs[i][1] == pairs[j][1]:
                            continue
                        else:
                            triples.append([pairs[i][0],pairs[j][0],pairs[i][2]*pairs[j][2]])
                sorted_triples = sorted(triples, key= lambda triple: triple[2], reverse=True)
                for triple in sorted_triples:
                    ind1 = triple[0]
                    ind2 = triple[1]
                    xmin1 = int(round(result1[ind1][0]*width_crop)) + xmin
                    xmax1 = int(round(result1[ind1][2]*width_crop)) + xmin
                    ymin1 = int(round(result1[ind1][1]*height_crop)) + ymin
                    ymax1 = int(round(result1[ind1][3]*height_crop)) + ymin
                    
                    xmin2 = int(round(result1[ind2][0]*width_crop)) + xmin
                    xmax2 = int(round(result1[ind2][2]*width_crop)) + xmin
                    ymin2 = int(round(result1[ind2][1]*height_crop)) + ymin
                    ymax2 = int(round(result1[ind2][3]*height_crop)) + ymin
                    
                    score = item[5]*triple[2]
                    
                    relationship = item[6]
                    label1 = result1[ind1][6]
                    label2 = result1[ind2][6]
                    triplet = ','.join([label1,label2,relationship])
                    print triplet
                    lis = []
                    if triplet in relationship_triplet:
                        print "triplet found"
                        lis = [f, label1, label2, relationship, score, xmin1, xmax1, ymin1, ymax1, xmin2, xmax2, ymin2, ymax2]
                        output.write(','.join([str(x) for x in lis])+'\n')
                        break
                    triplet = ','.join([label2,label1,relationship])
                    if triplet in relationship_triplet:
                        print "triplet found"
                        lis = [f, label2, label1, relationship, score, xmin2, xmax2, ymin2, ymax2, xmin1, xmax1, ymin1, ymax1]
                        output.write(','.join([str(x) for x in lis])+'\n')
                        break
                    
    output.close()            
                

def parse_args():
    '''parse args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--labelmap_region',
                        default='data/VOC0712/labelmap_voc.prototxt')
    parser.add_argument('--labelmap_crop',
                        default='data/VOC0712/labelmap_voc.prototxt')
    parser.add_argument('--proto_region',
                        default='models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt')
    parser.add_argument('--proto_crop',
                        default='models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt')
    parser.add_argument('--image_resize', default=300, type=int)
    parser.add_argument('--model_region',
                        default='modelsVGGNet/VOC0712/SSD_300x300/'
                        'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')
    parser.add_argument('--model_crop',
                        default='modelsVGGNet/VOC0712/SSD_300x300/'
                        'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel')

    parser.add_argument('--image_dir')
    parser.add_argument('--output_file')
    parser.add_argument('--relationship_file')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
