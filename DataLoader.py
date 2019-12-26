"""
Main DataLoader for the Triplet Imagenet Project
Created By:
Mohammad Mushfequr Rahman
mohammadmushfequr.rahman@ontariotechu.net
"""

# Initialize Imports:
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import imutils
from PIL import Image
import os



class InterpolationData():
    def __init__(self, data_1,data_2,seed, width, height, test=False, transforms=None,step_size=1):
        self.data_1 = data_1
        self.data_2 = data_2
        self.step_size = step_size
        self.width = width
        self.height = height
        
    def __len__(self):
        return len(self.data_1)//2

    def __getitem__(self, idx):
        print("Idx: ",idx)
        print(self.data_1[idx])
        prev_frame = idx
        if self.data_1[idx][0]%10 == 0:
            if(idx==0):
                prev_frame = idx+1
            else:
                prev_frame = idx-1

        else:
            prev_frame = idx+1

        range = []
        ex_1 = 0
        ex_2 = 0
        if prev_frame>idx:
            range= np.arange(self.data_1[idx][0],self.data_1[prev_frame][0],self.step_size)
            ex_1 = idx
            ex_2 = prev_frame
        else:
            range = np.arange(self.data_1[prev_frame][0],self.data_1[idx][0], self.step_size)
            ex_1=prev_frame
            ex_2 = idx

        first_frame = np.divide(cv2.imread(self.data_1[ex_1][1]).astype(np.float32), 255)
        in_between_frames = []
        for i in range:
            in_between_frames.append(np.divide(cv2.imread(self.data_2[i][1]).astype(np.float32), 255))

        end_frame =   np.divide(cv2.imread(self.data_1[ex_2][1]).astype(np.float32), 255)





        """
        Plot out the images in a pyplot. 
        """
        #set this to true if you wish to see transformations
        show = False
        if (show== True):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(first_frame)
            ax1.set_title("First Frame")
            in_1 = cv2.hconcat(in_between_frames)
            ax2.imshow(in_1)
            ax2.set_title("Inbetween Frame")
            ax3.imshow(end_frame)
            ax3.set_title("Last Frame")
            plt.show()



        for i,img in enumerate(in_between_frames):
            in_between_frames[i]=cv2.resize(img,(self.width,self.height))


        img1 = torch.tensor(cv2.resize(first_frame, (self.width, self.height)))
        img2 = torch.tensor(in_between_frames)
        img3 = torch.tensor(cv2.resize(end_frame, (self.width, self.height)))

        # nameList = [int(self.data[idx][0][2]), int(self.data[idx][1][2])]
        # fileNums = torch.tensor(nameList, dtype=torch.float32)

        return {'A': img1, 'P': img2, 'N': img3}




def Read_From_Path(PATH):
    """

    :param PATH:
    :param out_file_name:
    :param in_between:
    :return: 2d list with frame_num X path_to_image
    """
    frame_count = 0
    class_count = 0
    frames = [frame for frame in os.listdir(PATH)]
    items = []


    for frame in frames:
        _,id = frame.split("_")
        path = os.path.join(PATH,(frame))
        id_1,_ = id.split(".")
        data_item = [int(id_1),path]

        items.append(data_item)

    items = sorted(items,key=lambda x: x[0])
    print(items)
    return items


def main():

    path_1 = "out_1/"
    path_2 = "out_2/"

    d1 = Read_From_Path(path_1)
    print("+++++++++Inbetween Frames++++++++++")
    d2 = Read_From_Path(path_2)


    train_loader = DataLoader(InterpolationData(d1,d2,1,105,105))


    for batch_i, batch_data in enumerate(train_loader):
        A, P, N= batch_data['A'], batch_data['P'], batch_data['N']
        # print("Classes: ", classes)
        print("A.shape:", A.shape)
        print("P.shape:", P.shape)
        print("N.shape:", N.shape)

        if ((batch_i + 1) % 10 == 0):
            break

    print("Done")


if __name__ == "__main__":
    main()
