import cv2
import numpy as py
import os.path
import os
import argparse 
parse=argparse.ArgumentParser(description="Detect image features")
parse.add_argument('dataset',help='add a images folder')
args=parse.parse_args()
path = os.path.join(args.dataset,'images')
for name in os.listdir(path):
  print os.path.join(args.dataset+'/'+name)
  image = cv2.imread(os.path.join(args.dataset+'/images/'+name),1)
  image_resize = cv2.resize(image,(0,0),fx=0.3,fy=0.3)
  cv2.imwrite(name,image_resize)
