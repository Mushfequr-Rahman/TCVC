import numpy as np
import cv2
import os
import argparse
from os import listdir
import sys

parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--video_path', required=True, help='Path of video to extract frames')
parser.add_argument('--output_path', required=True, help='output folder for frames')
parser.add_argument('--second_out', required=True, help='Second Output file')
parser.add_argument('--num_frames', type=int, default=10, help='num of frames to interpolate between ')

opt = parser.parse_args()
print(opt)

video_path = opt.video_path

output_path = opt.output_path
out_2 = opt.second_out
n = opt.num_frames
counter = 0

look_ahead = False 
currentframe = 0
cap = cv2.VideoCapture(video_path)

while(True):
        
    ret, frame = cap.read()
    frame_name = os.path.join(output_path,'frame_'+ str(counter).zfill(5) + '.jpg')
    second_name = os.path.join(out_2,'frame_'+ str(counter).zfill(5) + '.jpg')
    
    if np.shape(frame) == ():
        break


    if (currentframe%n==0):
        look_ahead = True      
        print('Creating...'+ frame_name)
        cv2.imwrite(frame_name, frame)
        counter += 1
        
    elif(look_ahead and currentframe!=1):
        look_ahead = False
        print('Creating...' + frame_name)
        cv2.imwrite(frame_name, frame)
        counter += 1
    else:
        look_ahead = False
        print('Creating...' + second_name)
        cv2.imwrite(second_name, frame)
        counter += 1
        
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    currentframe +=1


cap.release()

cv2.destroyAllWindows()
    
