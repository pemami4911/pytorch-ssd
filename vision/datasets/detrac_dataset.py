"""
This file provides a dataset class for working with the UA-detrac tracking dataset.
Provides:
    - plotting of 2D bounding boxes
    - training/testing loader mode (random images from across all tracks) using __getitem__()
    - track mode - returns a single image, in order, using __next__()
    
It is assumed that image dir is a directory containing a subdirectory for each track sequence
Label dir is a directory containing a bunch of label files
"""

import os
import numpy as np

import cv2
from PIL import Image
from torch.utils import data
import xml.etree.ElementTree as ET

from vision.datasets.detrac_plot_utils import pil_to_cv, plot_bboxes_2d


class UADETRAC_Detection_Dataset(data.Dataset):
    """
    Creates an object for referencing the UA-Detrac 2D object tracking dataset
    
    For our multi-task problem, each minibatch should contain
     (Detection) 
     1. Pre-processed images
     2. bounding boxes
    """
    
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None, parse_tracks=False):
        """ initializes object. By default, the first track (cur_track = 0) is loaded 
        such that next(object) will pull next frame from the first track"""
        self.transform = transform
        self.target_transform = target_transform
        # stores files for each set of images and each label
        dir_list = next(os.walk(image_dir))[1]
        sequence_img_list = [os.path.join(image_dir,item) for item in dir_list]
        #sequence_label_list = [os.path.join(label_dir,item) for item in os.listdir(label_dir)]
        sequence_img_list.sort()
        #sequence_label_list.sort()
        
        # parse and store all labels and image names in a list such that
        # all_data[i] returns dict with image and bounding boxes and class
        self.all_data = []
        self.all_tracks = []
        self.sequence_list = sequence_img_list
        self.sequence_index_map = {}
            
        frame_idx = 0
        self.class_names = ('Background', 'Bus', 'Car', 'Truck')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.ua_detrac_class_map = {
            'Sedan': 2, 'Suv': 2, 'Van': 2, 'Taxi': 2, 'Bus': 1, 'Truck': 3, 'Truck-Box-Large': 3, 'Hatchback': 2, 'Police': 2, 'MiniVan': 2, 'Truck-Box-Med': 3, 'Truck-Box-Small': 3,
            'Truck-Flatbed': 3, 'Truck-Pickup': 3, 'Truck-Util': 3
        }
        
        for i in range(len(sequence_img_list)):
            if i > 0:
                self.sequence_index_map[frame_idx] = sequence_img_list[i-1]

            images = [os.path.join(sequence_img_list[i],frame) for frame in os.listdir(sequence_img_list[i])]
            images.sort() 
            seq_name = sequence_img_list[i].split('/')[-1] + '_v3.xml'
            labels,metadata = self.parse_labels(os.path.join(label_dir,seq_name), num_images=len(images))
            # each image in the sequence
            for j in range(len(images)):               
                if labels[j][0] == 'pass':
                    continue
                else:
                    out_dict = {
                            'image':images[j],
                            'label':labels[j],
                            'frame_id': j+1
                            }
                    self.all_data.append(out_dict)
                frame_idx += 1

            if parse_tracks:
                seq_tracks = self.parse_tracks(os.path.join(label_dir,seq_name))
                self.all_tracks += [seq_tracks]

        self.sequence_index_map[frame_idx] = sequence_img_list[-1]
        self.total_num_frames = len(self.all_data)
        
        
    def _read_image(self, image_id):
        cur = self.all_data[image_id]
        image = cv2.imread(cur['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, cur

    def _get_sample(self, index):
        image, cur = self._read_image(index)
        label = cur['label']
        frame_id = cur['frame_id']
        boxes = []
        labels = []
        for frame in label:
            boxes += [frame['bbox']]
            labels += [frame['class']]
        boxes = np.array(boxes)
        labels = np.array(labels)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        for k,v in self.sequence_index_map.items():
            if index >= k:
                continue
            return image, boxes, labels, frame_id, v

    def get_image(self, i):
        image, boxes, labels, frame_id, seq = self._get_sample(i)
        return image, frame_id, seq

    def __len__(self):
        """ returns total number of frames in all tracks"""
        return self.total_num_frames
    
    def __getitem__(self,index):
        """ returns item indexed from all frames in all tracks"""
        image, boxes, labels, _, _ = self._get_sample(index)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels
    
    def parse_labels(self,label_file, num_images):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        frames = root.getchildren()
        
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])]).astype('float32')
            ignored_regions.append(box)
        frames = frames[2:]
        
        # rest are bboxes
        all_boxes = []
        cur_id = 1 
        for frame in frames:
            frame_id = int(frame.attrib['num'])
            while frame_id != cur_id:
                all_boxes += [['pass']]
                cur_id += 1
            cur_id += 1

            frame_boxes = []
            boxids = frame.getchildren()[0].getchildren()
            for boxid in boxids:
                data = boxid.getchildren()
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                det_dict = {
                        'id':int(boxid.attrib['id']),
                        'class':self.ua_detrac_class_map[stats['vehicle_type']],
                        #'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox
                        }
                
                frame_boxes.append(det_dict)
            all_boxes.append(frame_boxes)
        if len(all_boxes) < num_images:
            for i in range(num_images - len(all_boxes)):
                all_boxes += [['pass']]

        sequence_metadata = {
                'sequence':seq_name,
                'seq_attributes':seq_attrs,
                'ignored_regions':ignored_regions
                }
        return all_boxes, sequence_metadata

    
    def parse_tracks(self,label_file):
        """
        Returns a set of metadata (1 per track) and a list of labels (1 item per
        frame, where an item is a list of dictionaries (one dictionary per object
        with fields id, class, truncation, orientation, and bbox
        """
        tree = ET.parse(label_file)
        root = tree.getroot()
        
        # get sequence attributes
        seq_name = root.attrib['name']
        
        # get list of all frame elements
        frames = root.getchildren()
        
        # first child is sequence attributes
        seq_attrs = frames[0].attrib
        
        # second child is ignored regions
        ignored_regions = []
        for region in frames[1]:
            coords = region.attrib
            box = np.array([float(coords['left']),
                            float(coords['top']),
                            float(coords['left']) + float(coords['width']),
                            float(coords['top'])  + float(coords['height'])]).astype('float32')
            ignored_regions.append(box)
        frames = frames[2:]
        
        all_tracks = {}
        for frame in frames:
            frame_id = int(frame.attrib['num'])
            boxids = frame.getchildren()[0].getchildren()
            for boxid in boxids:
                data = boxid.getchildren()
                coords = data[0].attrib
                stats = data[1].attrib
                bbox = np.array([float(coords['left']),
                                float(coords['top']),
                                float(coords['left']) + float(coords['width']),
                                float(coords['top'])  + float(coords['height'])])
                
                id = int(boxid.attrib['id'])
        
                det_dict = {
                        'class':self.ua_detrac_class_map[stats['vehicle_type']],
                        #'color':stats['color'],
                        'orientation':float(stats['orientation']),
                        'truncation':float(stats['truncation_ratio']),
                        'bbox':bbox,
                        'seq_name': seq_name,
                        'frame_id': frame_id
                }
                
                if id in all_tracks:
                    all_tracks[id] += [det_dict]
                else:
                    all_tracks[id] = [det_dict]

        return all_tracks 
