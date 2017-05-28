import cv2
import numpy as np
srcimage=cv2.imread('../berlin/images/01.jpg',1)
cv2.imshow('image',srcimage)
cv2.waitKey(0)
roi=srcimage[20:150,170:350]
srcimage[0:130,0:180]=roi
cv2.imshow('image',srcimage)
cv2.waitKey(0)
