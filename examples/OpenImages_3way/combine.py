import pandas as pd
import sys
import os
from PIL import Image

is_attr = pd.read_csv(sys.argv[1])
is_attr = is_attr[is_attr['Confidence'] >= 0.1]
rest_attr = pd.read_csv(sys.argv[2])
rest_attr = rest_attr[rest_attr['Confidence'] >= 0.001]
image_dir = sys.argv[3]
output_file = open(sys.argv[4],'w')

count = 0
for _,_,files in os.walk(image_dir):
        for f in files:
            img = Image.open(image_dir+'/'+f)
            width, height = img.size
            count+=1
            if count%1000 == 0:
                print count
            result_is = is_attr[is_attr['ImageID'] == f]
            result_rest = rest_attr[rest_attr['ImageID'] == f]
            entry = ''
            for _, row in result_is.iterrows():
                    arr = [row.Confidence, row.Label, max(0,float(row.xmin)/width), max(0,float(row.ymin)/height), min(1,float(row.xmax)/width), min(1,float(row.ymax)/height), row.Attr, max(0,float(row.xmin)/width), max(0,float(row.ymin)/height), min(1,float(row.xmax)/width), min(1,float(row.ymax)/height), "is"]
                    entry += ' ' + ' '.join(str(x) for x in arr)
            for _, row in result_rest.iterrows():
                    arr = [row.Confidence, row.Label1, max(0,float(row.xmin1)/width), max(0,float(row.ymin1)/height), min(1,float(row.xmax1)/width), min(1,float(row.ymax1)/height), row.Label2, max(0,float(row.xmin2)/width), max(0,float(row.ymin2)/height), min(1,float(row.xmax2)/width), min(1,float(row.ymax2)/height), row.Relationship]
                    entry += ' ' + ' '.join(str(x) for x in arr)
            entry = f[:-4]+','+entry+'\n'
            output_file.write(entry)
output_file.close()
