# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:43:29 2023

@author: Tahasanul Abraham
"""

# %% Initialization of Libraries and Directory

# Standard library imports
from xml.etree import ElementTree as et
from queue import Queue
from threading import Thread, Lock
from pathlib import Path

# Third-party imports
import torch
import torchvision
import numpy as np
from PIL import Image

# Local application imports
from lib.utils import collate_fn

#%% Worker

class MultiThread_Worker( Thread ):
    def __init__( self, tasks ):
        Thread.__init__( self )
        self.tasks = tasks
        self.daemon = True
        self.lock = Lock( )
        self.start( )

    def run( self ):
        while True:
            func, args, kargs = self.tasks.get( )
            try:
                if func.lower( ) == "terminate":
                    break
            except:                
                try: 
                    with self.lock:
                        func( *args, **kargs )
                except Exception as exception: 
                    print( exception )
            self.tasks.task_done( )

#%% ThreadPool

class MultiThread_ThreadPool:
    def __init__( self, num_threads, num_queue=None ):
        if num_queue is None or num_queue < num_threads:
            num_queue = num_threads
        self.tasks = Queue( num_queue )
        self.threads = num_threads
        for _ in range( num_threads ): MultiThread_Worker( self.tasks )
        
    def terminate( self ):
        self.wait_completion( )
        for _ in range( self.threads ): self.add_task( "terminate" )
        return None

    def add_task( self, func, *args, **kargs ):
        self.tasks.put( ( func, args, kargs ) )

    def wait_completion( self ):
        self.tasks.join( )
        
    def is_alive( self ):
        if self.tasks.unfinished_tasks == 0:
            return False
        else:
            return True
#%% ModelAllDataset

class ModelAllDataset( torchvision.datasets.DatasetFolder ):
    def __init__( self, path, all_classes=None, transforms=None, retry=10, loader=Image.open, extensions=( "jpg","png" ) ):
        if not loader is None:
            super( ).__init__( path, loader=loader, extensions=extensions )
        else:            
            super( ).__init__( path, loader=loader, extensions=extensions )
        self.retry = retry
        self.__set_classes__( all_classes )
        
        if transforms is None:
            self.fixed_transforms = torchvision.transforms.Compose( 
                                                                        [
                                                                            torchvision.transforms.ToTensor( )
                                                                        ]
                                                                   )
        else:
            self.fixed_transforms = transforms
        
        self.training_data_subset = None
        self.training_data_subset_indexes = []
        
        self.testing_data_subset = None
        self.testing_data_subset_indexes = []
        
        self.validation_data_subset = None
        self.validation_data_subset_indexes = []
        
        self.masked = False
        
    def __getitem__( self, index ):
        path, class_to_idx = self.samples[index]
        sample = self.loader( path )
        sample = self.__remove_transparency__( sample )
        
        boxes, labels = self.__get_xml_info__( path, self.classes )            
        
        # bounding box to tensor
        boxes = torch.as_tensor( boxes, dtype=torch.float32 )
        # area of the bounding boxes
        area = ( boxes[:, 3] - boxes[:, 1] ) * ( boxes[:, 2] - boxes[:, 0] )
        # no crowd instances
        iscrowd = torch.zeros( ( boxes.shape[0], ), dtype=torch.int64 )
        # labels to tensor
        labels = torch.as_tensor( labels, dtype=torch.int64 )
        # class_to_idx to tensor
        class_to_idx = torch.as_tensor( class_to_idx, dtype=torch.int64 )
        # prepare the final `target` dictionary
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["class_to_idx"] = class_to_idx
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor( [index] )
        
        if self.masked is True:
            
            masks = self.__get_masks__( path, self.classes )
            
            if not len( masks ) == 0:   
                
                # masks to tensor
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                target["masks"] = masks
        
        return self.fixed_transforms( sample ), target
    
    def __get_masks__( self, path, all_classes ):
        masks = []
        p = Path(path)
        annot_file_path = Path(p.with_suffix("").as_posix() + ".mask")
        
        try:
            sample = self.loader( annot_file_path )
            sample = self.__remove_transparency__( sample )
            
            # convert the PIL Image into a numpy array
            mask = np.array(sample)
            # instances are encoded as different colors
            obj_ids = np.unique(mask)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = mask == obj_ids[:, None, None]
            
        except:
            pass
        
        return masks
        
    
    def __get_xml_info__( self, path, all_classes ):
        boxes = []
        labels = []
        p = Path(path)
        annot_file_path = Path(p.with_suffix("").as_posix() + ".xml")
        
        if annot_file_path.exists() is True:
            tree = et.parse( annot_file_path )
            root = tree.getroot( )            
            
            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall( "object" ):
                try:
                    # map the current object name to `classes` list to get...
                    # ... the label index and append to `labels` list
                    # try:
                    #     labels.append( all_classes.index( member.find( "name" ).text ) )
                    # except:
                    #     print (member.find( "name" ).text )
                    
                    labels.append( all_classes.index( member.find( "name" ).text ) )
                    
                    # xmin = left corner x-coordinates
                    xmin = int( float( member.find( "bndbox" ).find( "xmin" ).text ) )
                    # xmax = right corner x-coordinates
                    xmax = int( float( member.find( "bndbox" ).find( "xmax" ).text ) )
                    # ymin = left corner y-coordinates
                    ymin = int( float( member.find( "bndbox" ).find( "ymin" ).text ) )
                    # ymax = right corner y-coordinates
                    ymax = int( float( member.find( "bndbox" ).find( "ymax" ).text ) )
                    
                    boxes.append( [xmin, ymin, xmax, ymax] )
                except:
                    pass
            
        return boxes, labels
        
    def __remove_transparency__( self, im, bg_colour=( 255, 255, 255 ) ):
        """
        Remove transparency of the alpha value of PIL images.
        Parameters
        ----------
        im : Image
            PIL image which should be converted to RBG mode.
        bg_colour : tuple, optional
            The background color which replace the transparency. The default is ( 255, 255, 255 ).
        Returns
        -------
        PIL.Image
            The new image if convertation is possible.
        """
        if im.mode in ( "RGBA", "LA" ) or ( im.mode == "P" and "transparency" in im.info ):
            alpha = im.convert( "RGBA" ).split( )[-1]
            bg = Image.new( "RGB", im.size, bg_colour + ( 255, ) )
            bg.paste( im, mask=alpha )
            return bg
        else:
            return im
        
    def __set_classes__( self, all_classes ):
        self.classes = all_classes
    
    def __get_selected_data__( self, index, selection, all_classes ):
        if self.samples[index][1] == self.class_to_idx[selection]:
            boxes, labels = self.__get_xml_info__( self.samples[index][0], all_classes )
            if not len( boxes ) == 0 and not len( labels ) == 0:
                
                if selection == "training":
                    self.training_data_subset_indexes.append( index )
                elif selection == "testing":
                    self.testing_data_subset_indexes.append( index )
                elif selection == "validation":
                    self.validation_data_subset_indexes.append( index )
            # else:
                
            #     print (Path(self.samples[index][0]))
            #     Path(self.samples[index][0]).unlink()
    
    def get_labels( self, settings, force=False ):
        if self.classes is None or force is True:
            self.classes = {}
            if settings["MULTI_THREADED_DATA_LOADING_NT"] is True:
                get_labels_pool = MultiThread_ThreadPool( 
                                                                settings["MULTI_THREADED_DATA_LOADING_WORKERS_NT"],
                                                                settings["MULTI_THREADED_DATA_LOADING_QUEUE_NT"]
                                                          )
                for index in range( len( self.samples ) ):
                    get_labels_pool.add_task( self.__extract_label__, self.samples[index][0] )                
                get_labels_pool.wait_completion( )
                del get_labels_pool            
            else:
                for index in range( len( self.samples ) ):
                    self.__extract_label__( self.samples[index][0] )
                    
            self.classes = list(self.classes.keys())
            
        return self.classes
                    
    def __extract_label__(self, sample_path):
        
        file_labels = []
        p = Path(sample_path)
        annot_file_path = Path(p.with_suffix("").as_posix() + ".xml")
        if annot_file_path.exists() is True:
            tree = et.parse( annot_file_path )
            root = tree.getroot( )            
            
            # box coordinates for xml files are extracted and corrected for image size given
            for member in root.findall( "object" ):
                try:                                        
                    file_labels.append( member.find( "name" ).text )
                except:
                    pass
                
        for label in file_labels:
            if not label in self.classes.keys():
                self.classes[label] = 0
            else:
                self.classes[label] += 1
    
    def get_training_data( self, all_classes, settings, force=False ):
        
        if settings["MASKED_CNN_GBL"] is True:
            self.masked = True
        else:
            self.masked = False
            
        if self.training_data_subset is None or force is True:
            if settings["MULTI_THREADED_DATA_LOADING_NT"] is True:
                training_data_subset_pool = MultiThread_ThreadPool( 
                                                                        settings["MULTI_THREADED_DATA_LOADING_WORKERS_NT"],
                                                                        settings["MULTI_THREADED_DATA_LOADING_QUEUE_NT"]
                                                                    )
                for index in range( len( self.samples ) ):
                    training_data_subset_pool.add_task( self.__get_selected_data__, index, "training", all_classes )                
                training_data_subset_pool.wait_completion( )
                del training_data_subset_pool            
                self.training_data_subset = torch.utils.data.Subset( self, self.training_data_subset_indexes )
            else:
                for index in range( len( self.samples ) ):
                    self.__get_selected_data__( index, "training", all_classes )
                self.training_data_subset = torch.utils.data.Subset( self, self.training_data_subset_indexes )
        
        return self.training_data_subset
    
    def get_testing_data( self, all_classes, settings, force=False ):
        
        if settings["MASKED_CNN_GBL"] is True:
            self.masked = True
        else:
            self.masked = False
            
        if self.testing_data_subset is None or force is True:
            if settings["MULTI_THREADED_DATA_LOADING_NT"] is True:
                testing_data_subset_pool = MultiThread_ThreadPool( 
                                                                        settings["MULTI_THREADED_DATA_LOADING_WORKERS_NT"],
                                                                        settings["MULTI_THREADED_DATA_LOADING_QUEUE_NT"]
                                                                    )
                for index in range( len( self.samples ) ):
                    testing_data_subset_pool.add_task( self.__get_selected_data__, index, "testing", all_classes )                
                testing_data_subset_pool.wait_completion( )
                del testing_data_subset_pool            
                self.testing_data_subset = torch.utils.data.Subset( self, self.testing_data_subset_indexes )
            else:
                for index in range( len( self.samples ) ):
                    self.__get_selected_data__( index, "testing", all_classes )
                self.testing_data_subset = torch.utils.data.Subset( self, self.testing_data_subset_indexes )
            
        return self.testing_data_subset
    
    def get_validation_data( self, all_classes, settings, force=False ):
        
        if settings["MASKED_CNN_GBL"] is True:
            self.masked = True
        else:
            self.masked = False
            
        if self.validation_data_subset is None or force is True:
            if settings["MULTI_THREADED_DATA_LOADING_NT"] is True:
                validation_data_subset_pool = MultiThread_ThreadPool( 
                                                                        settings["MULTI_THREADED_DATA_LOADING_WORKERS_NT"],
                                                                        settings["MULTI_THREADED_DATA_LOADING_QUEUE_NT"]
                                                                    )
                for index in range( len( self.samples ) ):
                    validation_data_subset_pool.add_task( self.__get_selected_data__, index, "validation", all_classes )                
                validation_data_subset_pool.wait_completion( )
                del validation_data_subset_pool            
                self.validation_data_subset = torch.utils.data.Subset( self, self.validation_data_subset_indexes )
            else:
                for index in range( len( self.samples ) ):
                    self.__get_selected_data__( index, "validation", all_classes )
                self.validation_data_subset = torch.utils.data.Subset( self, self.validation_data_subset_indexes )
            
        return self.validation_data_subset
    

    
#%% ModelAllDataloader

class ModelAllDataloader( torch.utils.data.DataLoader ):
    def __init__( self, dataset, batch_size=25, shuffle=False, num_workers=5, collate_fn=collate_fn ):        
        super( ).__init__( dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn )
        
