# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

#from cityscapesscripts.preparation.json2labelImg import json2labelImg
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)

import json
import os
from PIL import Image
from PIL import ImageDraw
from abc import ABCMeta, abstractmethod
import locale
import datetime
from collections import namedtuple


Point = namedtuple('Point', ['x', 'y'])

class CsObjectType():
    """Type of an object"""
    POLY = 1  # polygon
    BBOX2D = 2  # bounding box
    BBOX3D = 3  # 3d bounding box
    IGNORE2D = 4  # 2d ignore region


class Camera(object):
    def __init__(
        self,
        fx,
        fy,
        u0,
        v0,
        sensor_T_ISO_8855,
        imgWidth=2048,
        imgHeight=1024):
        self.fx = fx
        self.fy = fy
        self.u0 = u0
        self.v0 = v0
        self.sensor_T_ISO_8855 = sensor_T_ISO_8855
        self.imgWidth = imgWidth
        self.imgHeight = imgHeight

class CsObject:
    """Abstract base class for annotation objects"""
    __metaclass__ = ABCMeta

    def __init__(self, objType):
        self.objectType = objType
        # the label
        self.label = ""

        # If deleted or not
        self.deleted = 0
        # If verified or not
        self.verified = 0
        # The date string
        self.date = ""
        # The username
        self.user = ""
        # Draw the object
        # Not read from or written to JSON
        # Set to False if deleted object
        # Might be set to False by the application for other reasons
        self.draw = True

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def fromJsonText(self, jsonText, objId=-1): pass

    @abstractmethod
    def toJsonText(self): pass

    def updateDate(self):
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.utf8')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'en_US')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'us_us.utf8')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'us_us')
        except Exception:
            pass
        self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Mark the object as deleted
    def delete(self):
        self.deleted = 1
        self.draw = False

class CsPoly(CsObject):
    """Class that contains the information of a single annotated object as polygon"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.POLY)
        # the polygon as list of points
        self.polygon = []
        # the object ID
        self.id = -1

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += '({},{}) '.format(p.x, p.y)
            else:
                polyText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[0].x, self.polygon[0].y,
                    self.polygon[1].x, self.polygon[1].y,
                    self.polygon[-2].x, self.polygon[-2].y,
                    self.polygon[-1].x, self.polygon[-1].y)
        else:
            polyText = "none"
        text = "Object: {} - {}".format(self.label, polyText)
        return text

    def fromJsonText(self, jsonText, objId=-1):
        self.id = objId
        self.label = str(jsonText['label'])
        self.polygon = [Point(p[0], p[1]) for p in jsonText['polygon']]
        if 'deleted' in jsonText.keys():
            self.deleted = jsonText['deleted']
        else:
            self.deleted = 0
        if 'verified' in jsonText.keys():
            self.verified = jsonText['verified']
        else:
            self.verified = 1
        if 'user' in jsonText.keys():
            self.user = jsonText['user']
        else:
            self.user = ''
        if 'date' in jsonText.keys():
            self.date = jsonText['date']
        else:
            self.date = ''
        if self.deleted == 1:
            self.draw = False
        else:
            self.draw = True

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['id'] = self.id
        objDict['deleted'] = self.deleted
        objDict['verified'] = self.verified
        objDict['user'] = self.user
        objDict['date'] = self.date
        objDict['polygon'] = []
        for pt in self.polygon:
            objDict['polygon'].append([pt.x, pt.y])

        return objDict


class CsBbox2d(CsObject):
    """Class that contains the information of a single annotated object as bounding box"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX2D)
        # the polygon as list of points
        self.bbox_amodal_xywh = []
        self.bbox_modal_xywh = []

        # the ID of the corresponding object
        self.instanceId = -1
        # the label of the corresponding object
        self.label = ""

    def __str__(self):
        bboxAmodalText = ""
        bboxAmodalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_amodal_xywh[0], self.bbox_amodal_xywh[1],  self.bbox_amodal_xywh[2],  self.bbox_amodal_xywh[3])

        bboxModalText = ""
        bboxModalText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format(
            self.bbox_modal_xywh[0], self.bbox_modal_xywh[1], self.bbox_modal_xywh[2], self.bbox_modal_xywh[3])

        text = "Object: {}\n - Amodal {}\n - Modal {}".format(
            self.label, bboxAmodalText, bboxModalText)
        return text

    def setAmodalBox(self, bbox_amodal):
        # sets the amodal box if required
        self.bbox_amodal_xywh = [
            bbox_amodal[0],
            bbox_amodal[1],
            bbox_amodal[2] - bbox_amodal[0],
            bbox_amodal[3] - bbox_amodal[1]
        ]

    # access 2d boxes in [xmin, ymin, xmax, ymax] format
    @property
    def bbox_amodal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_amodal_xywh[0],
            self.bbox_amodal_xywh[1],
            self.bbox_amodal_xywh[0] + self.bbox_amodal_xywh[2],
            self.bbox_amodal_xywh[1] + self.bbox_amodal_xywh[3]
        ]

    @property
    def bbox_modal(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_modal_xywh[0],
            self.bbox_modal_xywh[1],
            self.bbox_modal_xywh[0] + self.bbox_modal_xywh[2],
            self.bbox_modal_xywh[1] + self.bbox_modal_xywh[3]
        ]

    def fromJsonText(self, jsonText, objId=-1):
        # try to load from cityperson format
        if 'bbox' in jsonText.keys() and 'bboxVis' in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['bbox']
            self.bbox_modal_xywh = jsonText['bboxVis']
        # both modal and amodal boxes are provided
        elif "modal" in jsonText.keys() and "amodal" in jsonText.keys():
            self.bbox_amodal_xywh = jsonText['amodal']
            self.bbox_modal_xywh = jsonText['modal']
        # only amodal boxes are provided
        else:
            self.bbox_modal_xywh = jsonText['amodal']
            self.bbox_amodal_xywh = jsonText['amodal']

        # load label and instanceId if available
        if 'label' in jsonText.keys() and 'instanceId' in jsonText.keys():
            self.label = str(jsonText['label'])
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['modal'] = self.bbox_modal_xywh
        objDict['amodal'] = self.bbox_amodal_xywh

        return objDict


class CsBbox3d(CsObject):
    """Class that contains the information of a single annotated object as 3D bounding box"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX3D)

        self.bbox_2d = None

        self.center = []
        self.dims = []
        self.rotation = []
        self.instanceId = -1
        self.label = ""
        self.score = -1.

    def __str__(self):
        bbox2dText = str(self.bbox_2d)

        bbox3dText = ""
        bbox3dText += '\n - Center (x/y/z) [m]: {}/{}/{}'.format(
            self.center[0], self.center[1],  self.center[2])
        bbox3dText += '\n - Dimensions (l/w/h) [m]: {}/{}/{}'.format(
            self.dims[0], self.dims[1],  self.dims[2])
        bbox3dText += '\n - Rotation: {}/{}/{}/{}'.format(
            self.rotation[0], self.rotation[1], self.rotation[2], self.rotation[3])

        text = "Object: {}\n2D {}\n - 3D {}".format(
            self.label, bbox2dText, bbox3dText)
        return text

    def fromJsonText(self, jsonText, objId=-1):
        # load 2D box
        self.bbox_2d = CsBbox2d()
        self.bbox_2d.fromJsonText(jsonText['2d'])

        self.center = jsonText['3d']['center']
        self.dims = jsonText['3d']['dimensions']
        self.rotation = jsonText['3d']['rotation']
        self.label = jsonText['label']
        self.score = jsonText['score']

        if 'instanceId' in jsonText.keys():
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['2d']['amodal'] = self.bbox_2d.bbox_amodal_xywh
        objDict['2d']['modal'] = self.bbox_2d.bbox_modal_xywh
        objDict['3d']['center'] = self.center
        objDict['3d']['dimensions'] = self.dims
        objDict['3d']['rotation'] = self.rotation

        return objDict

    @property
    def depth(self):
        # returns the BEV depth
        return np.sqrt(self.center[0]**2 + self.center[1]**2).astype(int)


class CsIgnore2d(CsObject):
    """Class that contains the information of a single annotated 2d ignore region"""

    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.IGNORE2D)

        self.bbox_xywh = []
        self.label = ""
        self.instanceId = -1

    def __str__(self):
        bbox2dText = ""
        bbox2dText += 'Ignore Region:  (x1: {}, y1: {}), (w: {}, h: {})'.format(
            self.bbox_xywh[0], self.bbox_xywh[1], self.bbox_xywh[2], self.bbox_xywh[3])

        return bbox2dText

    def fromJsonText(self, jsonText, objId=-1):
        self.bbox_xywh = jsonText['2d']

        if 'label' in jsonText.keys():
            self.label = jsonText['label']

        if 'instanceId' in jsonText.keys():
            self.instanceId = jsonText['instanceId']

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['2d'] = self.bbox_xywh

        return objDict

    @property
    def bbox(self):
        """Returns the 2d box as [xmin, ymin, xmax, ymax]"""
        return [
            self.bbox_xywh[0],
            self.bbox_xywh[1],
            self.bbox_xywh[0] + self.bbox_xywh[2],
            self.bbox_xywh[1] + self.bbox_xywh[3]
        ]

    # Extend api to be compatible to bbox2d
    @property
    def bbox_amodal_xywh(self):
        return self.bbox_xywh

    @property
    def bbox_modal_xywh(self):
        return self.bbox_xywh


class Annotation:
    """The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox2d)"""

    # Constructor
    def __init__(self, objType=CsObjectType.POLY):
        # the width of that image and thus of the label image
        self.imgWidth = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []
        # the camera calibration
        self.camera = None
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects = []
        # load objects
        if self.objectType != CsObjectType.IGNORE2D:
            for objId, objIn in enumerate(jsonDict['objects']):
                if self.objectType == CsObjectType.POLY:
                    obj = CsPoly()
                elif self.objectType == CsObjectType.BBOX2D:
                    obj = CsBbox2d()
                elif self.objectType == CsObjectType.BBOX3D:
                    obj = CsBbox3d()
                obj.fromJsonText(objIn, objId)
                self.objects.append(obj)

        # load ignores
        if 'ignore' in jsonDict.keys():
            for ignoreId, ignoreIn in enumerate(jsonDict['ignore']):
                obj = CsIgnore2d()
                obj.fromJsonText(ignoreIn, ignoreId)
                self.objects.append(obj)

        # load camera calibration
        if 'sensor' in jsonDict.keys():
            self.camera = Camera(fx=jsonDict['sensor']['fx'],
                                    fy=jsonDict['sensor']['fy'],
                                    u0=jsonDict['sensor']['u0'],
                                    v0=jsonDict['sensor']['v0'],
                                    sensor_T_ISO_8855=jsonDict['sensor']['sensor_T_ISO_8855'])

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)

        return jsonDict

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())



Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

name2label      = { label.name    : label for label in labels           }

import sys

def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    sys.exit(-1)

def createLabelImage(annotation, encoding, outline=None):
    # the size of the image
    size = ( annotation.imgWidth , annotation.imgHeight )

    # the background
    if encoding == "ids":
        background = name2label['unlabeled'].id
    elif encoding == "trainIds":
        background = name2label['unlabeled'].trainId
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw( labelImg )

    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # If the object is deleted, skip it
        if obj.deleted:
            continue

        # If the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        if ( not label in name2label ) and label.endswith('group'):
            label = label[:-len('group')]

        if not label in name2label:
            printError( "Label '{}' not known.".format(label) )

        # If the ID is negative that polygon should not be drawn
        if name2label[label].id < 0:
            continue

        if encoding == "ids":
            val = name2label[label].id
        elif encoding == "trainIds":
            val = name2label[label].trainId
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:
                drawer.polygon( polygon, fill=val, outline=outline )
            else:
                drawer.polygon( polygon, fill=val )
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    return labelImg


def json2labelImg(inJson,outImg,encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    labelImg   = createLabelImage( annotation , encoding )
    labelImg.save( outImg )


def convert_json_to_label(json_file):
    label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
    json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('cityscapes_path', help='cityscapes data path')
    parser.add_argument('--gt-dir', default='gtFine', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=1, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cityscapes_path = args.cityscapes_path
    out_dir = args.out_dir if args.out_dir else cityscapes_path
    mkdir_or_exist(out_dir)

    gt_dir = osp.join(cityscapes_path, args.gt_dir)

    poly_files = []
    for poly in scandir(gt_dir, '_polygons.json', recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    if args.nproc > 1:
        track_parallel_progress(convert_json_to_label, poly_files, args.nproc)
    else:
        track_progress(convert_json_to_label, poly_files)

    split_names = ['val']

    for split in split_names:
        filenames = []
        for poly in scandir(
                osp.join(gt_dir, split), '_polygons.json', recursive=True):
            filenames.append(poly.replace('_gtFine_polygons.json', ''))
        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)


if __name__ == '__main__':
    main()#