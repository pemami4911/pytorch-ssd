import os
import numpy as np

import cv2
from PIL import Image
from torch.utils import data
import glob


class UADETRAC_ReID_Dataset(data.Dataset):
    
    def __init__(self, image_dir, reid_file, transform=None):
        self.transform = transform
        self.reid_file = reid_file
        # stores files for each set of images
        dir_list = next(os.walk(image_dir))[1]
        sequence_img_list = [os.path.join(image_dir,item) for item in dir_list]
        sequence_img_list.sort()

        self.all_images = {}
        self.sequence_list = sequence_img_list
        self.class_names = ('Background', 'Bus', 'Car', 'Truck')

        for i in range(len(sequence_img_list)):
            seq_name = sequence_img_list[i].split('/')[-1]
            images = []
            self.all_images[seq_name] = {}
            for frame in os.listdir(sequence_img_list[i]):
                images += [os.path.join(sequence_img_list[i], frame)]
                self.all_images[seq_name][frame] = images[-1]
        self.reid_labels = []
        with open(self.reid_file, 'r') as f:
            for line in f:
                self.reid_labels += [line]

    def __len__(self):
        return len(self.reid_labels)
    
    def __getitem__(self,index):
        label = self.reid_labels[index]
        data = label.split(' ')
        seq = data[0]
        frame1 = data[1]
        frame2 = data[2]
        image1 = self.all_images[seq]['img' + str(frame1).zfill(5) + '.jpg']
        image2 = self.all_images[seq]['img' + str(frame2).zfill(5) + '.jpg']
        
        image1 = cv2.cvtColor(cv2.imread(image1), cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(cv2.imread(image2), cv2.COLOR_BGR2RGB)

        bbox1 = np.array([[ float(data[3]), float(data[4]), float(data[5]), float(data[6]) ]])
        bbox2 = np.array([[float(data[7]), float(data[8]), float(data[9]), float(data[10]) ]])
        if self.transform:
            image1, bbox1, _ = self.transform(image1, bbox1, None)
            image2, bbox2, _ = self.transform(image2, bbox2, None)
        return image1, bbox1, image2, bbox2


if __name__ == '__main__':
    data = UADETRAC_ReID_Dataset('/data/pemami/ua-detrac/Insight-MVT_Annotation_Val', '/data/pemami/ua-detrac/re_id_val.txt')
    print(data[0])


