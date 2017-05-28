#from opensfm import dataset
from opensfm import features
from opensfm import context
import numpy as np
import time
import logging
import argparse
import os.path,sys
import cv2
import pickle
#import json
#import matplotlib.pyplot as plt
PrimaryNum=500
MIN_MATCH=10

def parse_para():
  
  path = os.path.join(args.dataset,'images')
  if os.path.isfile(os.path.join(args.dataset,'imageslist.txt')):
    os.remove(os.path.join(args.dataset,'imageslist.txt'))
  for name in os.listdir(path):
    if is_image_file(name):
      with open(os.path.join(args.dataset,'imageslist.txt'), 'a') as fout:
        fout.write(args.dataset + '/images/' + name+'\n')

def read_image_list(L=None):
  if os.path.isfile(os.path.join(args.dataset,'imageslist.txt')):
    with open(os.path.join(args.dataset,'imageslist.txt')) as fin:
      for line in fin:
        #L.append(line)
        L.append(line.strip('\n'))
    return True
  else:
    parse_para()
    return False

def is_image_file(name):
  return name.split('.')[-1].lower() in {'jpg', 'jpeg', 'png', 'tif'}

def normalize_features(points,desc,colors,shapex, shapey):
  size=max(shapex,shapey)
  p = np.empty((len(points), 2))
  p[:, 0] = (points[:, 0] + 0.5 - shapex / 2.0) / size
  p[:, 1] = (points[:, 1] + 0.5 - shapey / 2.0) / size
  return p

def save_feature(image,points, features, color):
  try:
    if not os.path.isdir(os.path.join(args.dataset, 'features')):
      feature_path = os.makedirs(os.path.join(args.dataset, 'features'))
    path = os.path.join(os.path.join(args.dataset, 'features'), '{}.features.pkl.gz'.format(image))
    with open(path, 'w') as fin:
      pickle.dump([points,features,color], fin)
      #pickle.dump(features, fin)
    return True
  except OSError:
    return False

def detect_feature(L):
  #TODO:first choose if there are feature file. If it has, no need to do anything.
  print (cv2.__version__)
  keypoints=[]
  descriptors=[]
  colors=[]
  for L_child in L:
    IMREAD_COLOR = cv2.IMREAD_COLOR if context.OPENCV3 else cv2.CV_LOAD_IMAGE_COLOR
    image = cv2.imread(L_child, IMREAD_COLOR)[:, :, ::-1]  # Turn BGR to RGB
    # image = cv2.imread(L[0],1)[:,:,::-1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print image_single
    # image_out=cv2.cvtColor(image,cv2.COLOR_B)
    detector = cv2.FeatureDetector_create('SIFT')
    descriptor = cv2.DescriptorExtractor_create('SIFT')
    detector.setDouble("contrastThreshold", 0.1)
    detector.setDouble('edgeThreshold', 10)
    points = detector.detect(gray)
    points, desc = descriptor.compute(gray, points)
    #show detect feature in non-normalized coordinate
    cv2.drawKeypoints(gray, points, gray, color=255, flags=1)
    cv2.imshow('gray', gray)
    cv2.waitKey(0)
    #points_four include four types of data
    points_four = np.array([(i.pt[0], i.pt[1], i.size, i.angle) for i in points])
    color_image = image[points_four[:,1].round().astype(int),(points_four[:,0].round().astype(int))]

    #normalize point coordinate system
    points_normal = normalize_features(points_four,desc,color_image,image.shape[1],image.shape[0])

    #order point according to size-->strong
    order = np.argsort(points_four[:,2])
    keypoints = points_normal[order,:]
    descriptors = desc[order,:]
    colors = color_image[order,:]
    print L_child.split('/')[-1]
    #test read write feature <start>
    # if L_child.split('/')[-1]=='02.jpg':
    #     with open('before.txt', 'w') as fout:
    #         json.dump(keypoints[0:PrimaryNum, :].tolist(), fout)
    # save_feature(L_child.split('/')[-1],keypoints[0:PrimaryNum,:],descriptors[0:PrimaryNum,:],colors[0:PrimaryNum,:])
    save_feature(L_child.split('/')[-1],keypoints,descriptors,colors)

def read_feature_file(L_name_single):
  with open(os.path.join(os.path.join(args.dataset,'features'),'{}.features.pkl.gz'.format(L_name_single))) as fout:
    feature_for_single = pickle.load(fout)
    return feature_for_single

def good_matches_store(pairs_two, matches):
  try:
    if not os.path.isdir(os.path.join(args.dataset, 'matches')):
      feature_path = os.makedirs(os.path.join(args.dataset, 'matches'))
    path = os.path.join(os.path.join(args.dataset, 'matches'), '{}and.{}.matches.pkl.gz'.format(list(pairs_two)[0],list(pairs_two)[1]))
    with open(path, 'w') as fin:
      pickle.dump(matches, fin)
      # pickle.dump(features, fin)
    return True
  except OSError:
    return False

def match_lowe(feature1,feature2):
  FLANN_INDEX_KDTREE = 1
  flann_params = dict(algorithm=FLANN_INDEX_KDTREE, tree=4)
  flann = cv2.flann_Index(feature1, flann_params)
  idx, dist = flann.knnSearch(feature2, 2, params={})
  good1 = dist[:, 0] < dist[:, 1] * 0.8 * 0.8
  matches_good1 = zip(idx[good1, 0], good1.nonzero()[0])
  #(a, b) = np.array(matches_good, dtype=int)
  flann = cv2.flann_Index(feature2, flann_params)
  idx, dist = flann.knnSearch(feature1, 2, params={})
  good2 = dist[:, 0] < dist[:, 1] * 0.8 * 0.8
  matches_good2 = zip(idx[good2, 0], good2.nonzero()[0])
  matches1 = [(a, b) for a, b in np.array(matches_good1, dtype=int)]
  matches2 = [(b, a) for a, b in np.array(matches_good2, dtype=int)]
  matches = set(matches1).intersection(matches2)
  return np.array(list(matches),dtype=int)

def match_features_2D(L_name):
  #first read pickle file, then get descriptor and points and color then match
  pairs=set()
  res={im: L_name for im in L_name}
  for key,value in res.items():
    for val_ls in value:
      if key < val_ls:
        pairs.add((key,val_ls))

  for pairs_two in pairs:
      print pairs_two
      point_pair0, feature_pair0, color_pair0 = read_feature_file(list(pairs_two)[0])
      point_pair1, feature_pair1, color_pair1 = read_feature_file(list(pairs_two)[1])

  # #test read and write code <start>
  # point02, feature02, color_02 = read_feature_file('02.jpg')
  # print type(point02.tolist())
  # with open('after.txt','w') as fout:
  #     json.dump(point02.tolist(),fout)
      matches_good = match_lowe(feature_pair0,feature_pair1)


      good_matches_store(pairs_two, matches_good)

  #store good feature
      im1 = cv2.imread(list(pairs_two)[0])
      im2 = cv2.imread(list(pairs_two)[1])




if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
  start=time.time()
  parse=argparse.ArgumentParser(description="Detect image features")
  parse.add_argument('dataset',help='add a images folder')
  args=parse.parse_args()  
  parse_para()
  L=list()
  L_name=list()
  read_image_list(L)

  for image_name in L:
    L_name.append(image_name.split('/')[-1])

  print(L_name)
  detect_feature(L)
  match_features_2D(L_name)