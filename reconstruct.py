import pyopengv
import cv2
import numpy as np
import os,os.path
import logging
import argparse
import time
import networkx as nx
from networkx.algorithms import bipartite
from itertools import combinations
import json

def load_tracks_graph(fileobj):
    g = nx.Graph()
    for line in fileobj:
        image, track, observation, x, y, R, G, B = line.split('\t')
        g.add_node(image, bipartite=0)
        g.add_node(track, bipartite=1)
        g.add_edge(image, track,
            feature=(float(x), float(y)),
            feature_id=int(observation),
            feature_color=(float(R), float(G), float(B)))
    return g
def tracks_and_image(graph):
    tracks=[]
    images=[]
    path_test = os.path.join(args.dataset, 'node.txt')
    if not os.path.isfile(path_test):
        open(path_test,'w').close()
    for n in graph.nodes(data=True):
        # with open(path_test,'a') as fout:
        #     fout.write('{}\n'.format(n))
        if n[1]['bipartite']==0:
            images.append(n[0])
        else:
            tracks.append(n[0])
    return tracks,images

def common_track(graph, im1, im2):
    tracks = []
    p1, p2 = [], []
    for track in graph[im1]:
        if track in graph[im2]:
            tracks.append(track)
            p1.append(graph[im1][track]['feature'])
            p2.append(graph[im2][track]['feature'])
    p1 = np.array(p1)
    p2 = np.array(p2)
    return tracks, p1, p2

def pairwise_reconstructability(common_tracks, homography_inliers):
    outliers = common_tracks - homography_inliers
    outlier_ratio = float(outliers) / common_tracks
    if outlier_ratio > 0.3:
        return common_tracks
    else:
        return 0

def compute_image_pair(graph,image_graph,config):
    '''compute every image pair and calculate a score for each pair'''
    pair = []
    score = []
    for im1, im2, weight in image_graph.edges(data=True):
        tracks, p1, p2 = common_track(graph, im1, im2)
        if len(tracks) >= 50:
            H, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 0.004)
            # r is len(tracks) or zero
            r = pairwise_reconstructability(len(tracks), inliers.sum())
            if r > 0:
                pair.append((im1,im2))
                score.append(r)
    order = np.argsort(score)[::-1]
    return [pair[o] for o in order]

def load_exif(img):
    path = os.path.join(args.dataset,'exif/'+img+'.exif')
    with open(path,'r') as fin:
        return json.load(fin)

def load_camera():
    path = os.path.join(args.dataset,'camera_models.json')
    with open(path, 'r') as fin:
        return json.load(fin)

def bearing_point(point, camera_inter):
    points = point.reshape((-1, 1, 2))
    cameraMatrix=np.array([[float(camera_inter['focal']), 0., 0.],
                           [0., float(camera_inter['focal']), 0.],
                           [0., 0., 1.]], dtype=np.float32)
    distCoeffs = np.array([float(camera_inter['k1']), float(camera_inter['k2']), 0., 0.], dtype=np.float32)
    bearing_b = cv2.undistortPoints(points, cameraMatrix, distCoeffs).reshape(-1, 2)
    shp = bearing_b.shape[:-1]+(1,)
    bearing_b3 = np.hstack((bearing_b, np.ones(shp)))
    normbearing = np.linalg.norm(bearing_b3, axis=1)[:, np.newaxis]
    return bearing_b3/normbearing


def two_view_reconstruction(p1, p2, camera_inter1, camera_inter2, threshold):
    '''First adjust p1 and p2, then use opengv to get 3D point with five-point algorithm'''
    b1 = bearing_point(p1, camera_inter1)
    b2 = bearing_point(p2, camera_inter2)
    '''6月5日 22:48'''

def bootstrap_reconstruction(graph, im1, im2):
    exif1=load_exif(im1)
    exif2=load_exif(im2)
    cameras=load_camera()
    camera_inter1 = cameras[exif1['camera']]
    camera_inter2 = cameras[exif2['camera']]
    track_boot, p1, p2 = common_track(graph, im1, im2)
    '''The threshold is defined according to the reprojection error pixel number'''
    threshold = 0.006
    R, t, inliers = two_view_reconstruction(p1, p2, camera_inter1, camera_inter2, threshold)




def grow_reconstruction(tracks_graph, reconstruction, remaining_image):
    pass

if __name__=='__main__':
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    start = time.time()
    parse = argparse.ArgumentParser(description="Detect image features")
    parse.add_argument('dataset', help='add a images folder')
    args = parse.parse_args()
    path_track = os.path.join(args.dataset,'tracks.csv')
    with open(path_track, 'r') as fin:
        tracks_graph = load_tracks_graph(fin)
    tracks_id, images = tracks_and_image(tracks_graph)
    remaining_image = set(images)
    reconstructions=[]
    image_graph = bipartite.weighted_projected_graph(tracks_graph,images)
    # for data in image_graph.edges(data=True):
    #     print data
    pair_images = compute_image_pair(tracks_graph,image_graph,0.04)
    for im1, im2 in pair_images:
        if im1 in remaining_image and im2 in remaining_image:
            reconstruction = bootstrap_reconstruction(tracks_graph, im1, im2)
            # returned value is not bool type, the reconstruction is not empty then the if is True
            if reconstruction:
                remaining_image.remove(im1)
                remaining_image.remove(im2)
                reconstruction = grow_reconstruction(tracks_graph, reconstruction, remaining_image)
                reconstructions.append(reconstruction)
                reconstructions = sorted(reconstructions, key=lambda x: -len(x['shots']))



