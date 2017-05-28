import numpy as np
import time
import logging
import argparse
import os.path,sys
import pickle
import cv2

def plot_matches(image1,image2,matches,points1,points2,feature1,feature2,name):
    image_matche = cv2.cvtColor(np.zeros((image1.shape[0], image1.shape[1] * 2), dtype=np.uint8),cv2.COLOR_GRAY2BGR)
    # ROI_left = image_matche[0:image1.shape[0], 0:image1.shape[1]]
    # ROI_right = image_matche[0:image1.shape[0], image1.shape[1]:image1.shape[1]*2]
    p1_original = denormalized_image_coordinates(points1,image1.shape[1],image1.shape[0])
    for p in p1_original:
        # print (p[0].astype(int),p[1].astype(int))
        draw_circle(image1, (p[0].astype(int),p[1].astype(int)))
    image_matche[0:image1.shape[0], 0:image1.shape[1]] = image1
    p2_original = denormalized_image_coordinates(points2, image2.shape[1], image2.shape[0])
    for p in p2_original:
        # print (p[0].astype(int),p[1].astype(int))
        draw_circle(image2, (p[0].astype(int),p[1].astype(int)))
    image_matche[0:image1.shape[0], image1.shape[1]:image1.shape[1] * 2]=image2
    for match in matches:
        draw_line(image_matche, (p1_original[match[0]][0].astype(int), p1_original[match[0]][1].astype(int)), (p2_original[match[1]][0].astype(int) + image1.shape[1],p2_original[match[1]][1].astype(int)))
    # cv2.imwrite('../matches/name', image_matche)
    cv2.imshow('image_matche',image_matche)
    cv2.waitKey(0)

def draw_circle(image, point, diameter=3, color=(0,0,255)):
    cv2.circle(image,point,diameter,color,thickness=-1)

def draw_line(image,point1,point2,color=(0,255,0)):
    cv2.line(image,point1,point2,color,thickness=2,lineType=8)

def denormalized_image_coordinates(norm_coords, width, height):
    size = max(width, height)
    p = np.empty((len(norm_coords), 2))
    p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
    p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
    return p
def read_feature_file(L_name_single):
  with open(os.path.join(os.path.join(args.dataset,'features'),'{}.features.pkl.gz'.format(L_name_single))) as fout:
    feature_for_single = pickle.load(fout)
    return feature_for_single


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot matches between images')
    parser.add_argument('dataset',
                        help='path to the dataset to be processed')
    args = parser.parse_args()
    path = os.path.join(args.dataset, 'matches')
    path_image = os.path.join(args.dataset, 'images')

    for name in os.listdir(path):
        #images = cv2.imread(name)
        print name
        #name.split('.')[0] is 01 or 02 the type
        feature_path1 = os.path.join(os.path.join(args.dataset,'features'),name.split('.')[0] + '.jpg.features.pkl.gz')
        feature_path2 = os.path.join(os.path.join(args.dataset,'features'),name.split('.')[2] + '.jpg.features.pkl.gz')
        with open(feature_path1) as fout:
             points1,features1,color1 = pickle.load(fout)

        with open(feature_path2) as fout:
             points2,features2,color2 = pickle.load(fout)

        #print feature_path1
        with open(os.path.join(os.path.join(args.dataset,'matches'),name)) as fout:
            matches_for_pair = pickle.load(fout)
        #print os.path.join(path_image, name.split('.')[0])
        image1 = cv2.imread(os.path.join(path_image, name.split('.')[0]+'.jpg'))
        image2 = cv2.imread(os.path.join(path_image, name.split('.')[2]+'.jpg'))
        a,b,c = image1.shape

        plot_matches(image1,image2,matches_for_pair,points1,points2,features1,features2,name.split('.')[0])
        print ('THere are {} matches in images named {} and {}'.format(len(matches_for_pair),name.split('.')[0],name.split('.')[2]))
