
# license removed for brevity
import rospy
import actionlib
import math
import os
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import argparse
import time
import blobconverter
import rospkg
import threading
from queue import Queue

from tf.transformations import *
from tf import TransformListener
import tf
from std_msgs.msg import String

import time
import json

learn = True
        
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1.0, self.color, 2, self.line_type)
    def drawContours(self, frame, points):
        cv2.drawContours(frame, [points], 0, self.bg_color, 6)
        cv2.drawContours(frame, [points], 0, self.color, 2)

class FaceRecognition:
    def __init__(self, db_path, name) -> None:
        self.read_db(db_path)
        self.databases=db_path
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True
        #self.name = "John"

    def cosine_distance(self, a, b):
        if a.shape != b.shape:
            raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b.T) / (a_norm * b_norm)

    def new_recognition(self, results):
        conf = []
        max_ = 0
        label_ = None
        for label in list(self.labels):
            for j in self.db_dic.get(label):
                conf_ = self.cosine_distance(j, results)
                if conf_ > max_:
                    max_ = conf_
                    label_ = label

        conf.append((max_, label_))
        name = conf[0] if conf[0][0] >= 0.5 else (conf[0][0], "UNKNOWN")
        # self.putText(frame, f"name:{name[1]}", (coords[0], coords[1] - 35))
        # self.putText(frame, f"conf:{name[0] * 100:.2f}%", (coords[0], coords[1] - 10))

        if name[1] == "UNKNOWN":
            self.create_db(results)
        return name

    def read_db(self, databases_path):
        self.labels = []
        for file in os.listdir(databases_path):
            filename = os.path.splitext(file)
            if filename[1] == ".npz":
                self.labels.append(filename[0])

        self.db_dic = {}
        for label in self.labels:
            with np.load(f"{databases_path}/{label}.npz") as db:
                self.db_dic[label] = [db[j] for j in db.files]

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 1, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 1, self.color, 1, self.line_type)

    def create_db(self, results):
        if self.name is None:
            if not self.printed:
                print("Wanted to create new DB for this face, but --name wasn't specified")
                self.printed = True
            return
        print('Saving face...')
        try:
            with np.load(f"{self.databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{self.databases}/{self.name}", *db_)
        self.adding_new = False


class VisionSystem():

    def __init__(self):
      self.openvinoVersion = "2021.4"
      print(dai.__version__)
      parser = argparse.ArgumentParser()
      parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")
      self.args = parser.parse_args() 
      rospack = rospkg.RosPack()
      self.packageDir = rospack.get_path('floyd')
          
      self.databases = self.packageDir+"/databases"
      print(self.databases)
      if not os.path.exists(self.databases):
          os.mkdir(self.databases)

      # Tiny yolo v3/4 label texts
      self.labelMap = [
        "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
        "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
      ]
      self.speakPub = rospy.Publisher("mycroft/speak", String, queue_size=10)

      self.syncNN = True
      self.createPipeline()
      
      self.device = dai.Device(self.pipeline)
      self.device.setLogLevel(dai.LogLevel.WARN)
      self.device.setLogOutputLevel(dai.LogLevel.WARN)
      # Output queues will be used to get the rgb frames and nn data from the outputs defined above
      self.previewQueue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
      self.detectionNNQueue = self.device.getOutputQueue(name="detections", maxSize=4, blocking=False)
      self.xoutBoundingBoxDepthMappingQueue = self.device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
      self.depthQueue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
      self.recCfgQ = self.device.getOutputQueue("face_rec_cfg_out", 4, False)
      self.arcQ = self.device.getOutputQueue("arc_out", 4, False)
      self.faceSpatialDataQueue = self.device.getOutputQueue(name="faceSpatialData", maxSize=4, blocking=False)
      self.faceSpatialConfigQueue =self.device.getOutputQueue(name="spatialFaceConfig", maxSize=4, blocking=False)
      self.faceSpatialConfigBB =self.device.getOutputQueue(name="spatialFaceConfigBB_out", maxSize=8, blocking=False)
      self.faceRecognitionBB =self.device.getOutputQueue(name="face_recBB_out", maxSize=8, blocking=False)
      
      self.displayFaceQueue = Queue(maxsize = 10)
      self.displayYoloQueue = Queue(maxsize = 10)
      
      calibData = self.device.readCalibration()
      print(f"RGB FOV {calibData.getFov(dai.CameraBoardSocket.RGB)}, Mono FOV {calibData.getFov(dai.CameraBoardSocket.LEFT)}")
      
      self.faceThread = threading.Thread(target=self.runFacePipeline)
      self.faceThread.daemon = True
      self.faceThread.start()

      self.objectThread = threading.Thread(target=self.runObjectPipeline)
      self.objectThread.daemon = True
      self.objectThread.start()

      self.displayThread = threading.Thread(target=self.runDisplay)
      self.displayThread.daemon = True
      self.displayThread.start()


      
    def createPipeline(self):
    # Create pipeline
      self.pipeline = dai.Pipeline()
      self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
    
      # Define sources and outputs
      self.camRgb = self.pipeline.create(dai.node.ColorCamera)
      self.camRgb.setIspScale(2, 3)
      self.yoloDetectionNetwork = self.pipeline.create(dai.node.YoloSpatialDetectionNetwork)
      
      # ImageManip that will crop the frame before sending it to the Face detection NN node
      self.face_det_manip = self.pipeline.create(dai.node.ImageManip)
      self.face_det_manip.initialConfig.setResize(300, 300)
      self.face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
      self.face_det_nn = self.pipeline.create(dai.node.MobileNetDetectionNetwork)
      self.face_det_nn.setConfidenceThreshold(0.5)
      self.face_det_nn.setBlobPath(str(blobconverter.from_zoo(
          name="face-detection-retail-0004",
          shaves=6,
          version=self.openvinoVersion
      )))
      # Link Face ImageManip -> Face detection NN node
      self.face_det_manip.out.link(self.face_det_nn.input)
      
      # Script node will take the output from the face detection NN as an input and set ImageManipConfig
      # to the 'age_gender_manip' to crop the initial frame
      self.script = self.pipeline.create(dai.node.Script)
      self.script.setProcessor(dai.ProcessorType.LEON_CSS)
      self.face_det_nn.out.link(self.script.inputs['face_det_in'])
      # We are only interested in timestamp, so we can sync depth frames with NN output
      self.face_det_nn.passthrough.link(self.script.inputs['face_pass'])
      
      with open(self.packageDir+"/src/floyd/script.py", "r") as f:
          self.script.setScript(f.read())
      
      # ImageManip as a workaround to have more frames in the pool.
      # cam.preview can only have 4 frames in the pool before it will
      # wait (freeze). Copying frames and setting ImageManip pool size to
      # higher number will fix this issue.
      
      self.camRgb.setPreviewSize(416, 416)
      self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
      self.camRgb.setInterleaved(False)
      self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
      
      self.copy_manip = self.pipeline.create(dai.node.ImageManip)
      self.camRgb.preview.link(self.copy_manip.inputImage)
      self.copy_manip.setNumFramesPool(20)
      self.copy_manip.setMaxOutputFrameSize(1072*1072*3)
      
      self.copy_manip.out.link(self.face_det_manip.inputImage)
      self.copy_manip.out.link(self.script.inputs['preview'])
      
      print("Creating Head pose estimation NN")
      self.headpose_manip = self.pipeline.create(dai.node.ImageManip)
      self.headpose_manip.initialConfig.setResize(60, 60)
      
      self.script.outputs['manip_cfg'].link(self.headpose_manip.inputConfig)
      self.script.outputs['manip_img'].link(self.headpose_manip.inputImage)
      
      self.headpose_nn = self.pipeline.create(dai.node.NeuralNetwork)
      self.headpose_nn.setBlobPath(str(blobconverter.from_zoo(
          name="head-pose-estimation-adas-0001",
          shaves=6,
          version=self.openvinoVersion
      )))
      self.headpose_manip.out.link(self.headpose_nn.input)
      
      self.headpose_nn.out.link(self.script.inputs['headpose_in'])
      self.headpose_nn.passthrough.link(self.script.inputs['headpose_pass'])
      
      print("Creating face recognition ImageManip/NN")
      
      self.face_rec_manip = self.pipeline.create(dai.node.ImageManip)
      #self.face_rec_manip.initialConfig.setResize(128, 128)
      self.face_rec_manip.initialConfig.setResize(96, 112)
      
      self.script.outputs['manip2_cfg'].link(self.face_rec_manip.inputConfig)
      self.script.outputs['manip2_img'].link(self.face_rec_manip.inputImage)
      
      self.face_rec_cfg_out = self.pipeline.create(dai.node.XLinkOut)
      self.face_rec_cfg_out.setStreamName('face_rec_cfg_out')
      self.script.outputs['manip2_cfg'].link(self.face_rec_cfg_out.input)
      
      self.face_rec_bb_out = self.pipeline.create(dai.node.XLinkOut)
      self.face_rec_bb_out.setStreamName('face_recBB_out')
      self.script.outputs['face_recBB'].link(self.face_rec_bb_out.input)
      
      # Only send metadata for the host-side sync
      # pass2_out = pipeline.create(dai.node.XLinkOut)
      # pass2_out.setStreamName('pass2')
      # pass2_out.setMetadataOnly(True)
      # script.outputs['manip2_img'].link(pass2_out.input)
      
      self.face_rec_nn = self.pipeline.create(dai.node.NeuralNetwork)
      # Removed from OMZ, so we can't use blobconverter for downloading, see here:
      # https://github.com/openvinotoolkit/open_model_zoo/issues/2448#issuecomment-851435301
      # name="face-reidentification-retail-0095",
      self.face_rec_nn.setBlobPath(str(blobconverter.from_zoo(
          name="Sphereface",
          shaves=6,
          version=self.openvinoVersion
      )))
      self.face_rec_manip.out.link(self.face_rec_nn.input)
      
      if False:
          self.xout_face = self.pipeline.createXLinkOut()
          self.xout_face.setStreamName('face')
          self.face_rec_manip.out.link(self.xout_face.input)
      
      self.arc_out = self.pipeline.create(dai.node.XLinkOut)
      self.arc_out.setStreamName('arc_out')
      self.face_rec_nn.out.link(self.arc_out.input)
      
      self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
      self.monoRight = self.pipeline.create(dai.node.MonoCamera)
      self.stereo = self.pipeline.create(dai.node.StereoDepth)
      self.stereo.setLeftRightCheck(True)
      self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
      #self.stereo.setOutputSize(640,400)
      
      self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
      self.xoutNN = self.pipeline.create(dai.node.XLinkOut)
      self.xoutBoundingBoxDepthMapping = self.pipeline.create(dai.node.XLinkOut)
      self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
      
      self.xoutRgb.setStreamName("rgb")
      self.xoutNN.setStreamName("detections")
      self.xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
      self.xoutDepth.setStreamName("depth")
      
      # Properties

      #camRgb.preview.link(face_manip.inputImage)
      #face_manip.out.link(face_nn.input)
      
      self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
      self.monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
      self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
      
      # setting node configs
      #self.stereo.initialConfig.setConfidenceThreshold(255)
      #self.stereo.setLeftRightCheck(True)
      #self.stereo.setSubpixel(False)
      self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
      
      self.spatialFaceCalc = self.pipeline.create(dai.node.SpatialLocationCalculator)
      self.spatialFaceCalc.setWaitForConfigInput(False)
      
      # Set initial config
      self.faceSpatialConfig = dai.SpatialLocationCalculatorConfigData()
      self.faceSpatialConfig.depthThresholds.lowerThreshold = 100
      self.faceSpatialConfig.depthThresholds.upperThreshold = 10000
      self.faceSpatialConfig.calculationAlgorithm = dai.SpatialLocationCalculatorAlgorithm.MIN
      topLeft = dai.Point2f(0.4, 0.4)
      bottomRight = dai.Point2f(0.6, 0.6)
      self.faceSpatialConfig.roi = dai.Rect(topLeft, bottomRight)
      self.spatialFaceCalc.initialConfig.addROI(self.faceSpatialConfig)
      
      self.script.outputs['spatialFaceConfig'].link(self.spatialFaceCalc.inputConfig)
      self.stereo.depth.link(self.spatialFaceCalc.inputDepth)
      self.faceSpatialData = self.pipeline.create(dai.node.XLinkOut)
      self.faceSpatialData.setStreamName("faceSpatialData")
      self.spatialFaceCalc.out.link(self.faceSpatialData.input)

      self.xoutSpatialFaceConfig = self.pipeline.create(dai.node.XLinkOut)
      self.xoutSpatialFaceConfig.setStreamName("spatialFaceConfig")
      self.script.outputs['spatialFaceConfig'].link(self.xoutSpatialFaceConfig.input)

      self.faceSpatialConfigBB_out = self.pipeline.create(dai.node.XLinkOut)
      self.faceSpatialConfigBB_out.setStreamName('spatialFaceConfigBB_out')
      self.script.outputs['spatialFaceConfigBB'].link(self.faceSpatialConfigBB_out.input)
      
      #spatialDetectionNetwork.setBlobPath(nnBlobPath)
      self.yoloDetectionNetwork.setBlobPath(str(blobconverter.from_zoo(
          name="yolov4_tiny_coco_416x416",
          zoo_type="depthai",
          shaves=6,
      )))
      
      #self.yoloDetectionNetwork.setBlobPath(self.packageDir+"/models/yolox_tiny.blob")
      #self.yoloDetectionNetwork.setNumInferenceThreads(2)
      self.yoloDetectionNetwork.setConfidenceThreshold(0.5)
      self.yoloDetectionNetwork.input.setBlocking(False)
      self.yoloDetectionNetwork.setBoundingBoxScaleFactor(0.5)
      self.yoloDetectionNetwork.setDepthLowerThreshold(100)
      self.yoloDetectionNetwork.setDepthUpperThreshold(5000)

      # Yolo specific parameters
      self.yoloDetectionNetwork.setNumClasses(80)
      self.yoloDetectionNetwork.setCoordinateSize(4)
      #self.yoloDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
      self.yoloDetectionNetwork.setAnchors(np.array([23,27, 37,58, 81,82, 81,82, 135,169, 344,319]))
      self.yoloDetectionNetwork.setAnchorMasks({ "side26": np.array([0,1,2]), "side13": np.array([3,4,5]) })
      self.yoloDetectionNetwork.setIouThreshold(0.35)
      
      # Linking
      self.monoLeft.out.link(self.stereo.left)
      self.monoRight.out.link(self.stereo.right)
      
      self.camRgb.preview.link(self.yoloDetectionNetwork.input)
      self.yoloDetectionNetwork.passthrough.link(self.xoutRgb.input)
      #spatialDetectionNetwork.passthrough.link(xoutRgb.input)
      
      self.yoloDetectionNetwork.out.link(self.xoutNN.input)
      self.yoloDetectionNetwork.boundingBoxMapping.link(self.xoutBoundingBoxDepthMapping.input)
      
      self.stereo.depth.link(self.yoloDetectionNetwork.inputDepth)
      self.yoloDetectionNetwork.passthroughDepth.link(self.xoutDepth.input)
    

    def runFacePipeline(self):
        firstTime = time.monotonic()
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        colorFace = (255, 0, 0)
        
        facerec = FaceRecognition(self.databases, self.args.name)
        lastPersonAnnouncement = 0
        lastObjectAnnouncement = 0

        while True:
            time.sleep(0.001)
   
            counter+=1
            current_time = time.monotonic()
            if (current_time - startTime) > 1 :
                fps = counter / (current_time - startTime)
                counter = 0
                startTime = current_time
                localTime = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
                print(f"{current_time-firstTime}, {localTime}")
    
    
            height = 720 #frame.shape[0]
            width  = 1280 #frame.shape[1]
            
            cfg = self.recCfgQ.tryGet()
            results = {}
            if cfg is not None:
                bbSpatial = self.faceSpatialConfigBB.get()
                
                rr = cfg.getRaw().cropConfig.cropRotatedRect
                arcIn = self.arcQ.get()
                center = (int(rr.center.x * width), int(rr.center.y * height))
                size = (int(rr.size.width * width), int(rr.size.height * height))
                rotatedRect = (center, size, rr.angle)
                points = np.int0(cv2.boxPoints(rotatedRect))
                features = np.array(arcIn.getFirstLayerFp16())
                conf, name = facerec.new_recognition(features)
                
                if time.time() - lastPersonAnnouncement > 10*60:
                   if name == "UNKNOWN":
                       self.speakPub.publish("Detected unknown person")
                   else:
                       self.speakPub.publish("Detected %s" % name)
                   lastPersonAnnouncement = time.time()
                
                result = {
                    'name': name,
                    'conf': conf,
                    'coords': center,
                    'points': points,
                    'ts': time.time(),
                }
                bb = json.loads(str(bbSpatial.getData(), 'utf-8'))
                spatialData = self.faceSpatialDataQueue.get().getSpatialLocations()
                for depthData in spatialData:
                    print(f"{name} at {(depthData.spatialCoordinates.x/1000):.2f}, {(depthData.spatialCoordinates.y/1000):.2f}, {(depthData.spatialCoordinates.z/1000):.2f}, {conf:.2f}, {bb['xmin']:.2f},{bb['ymin']:.2f} / {bb['xmax']:.2f}, {bb['ymax']:.2f}")
                    result['x']=depthData.spatialCoordinates.x
                    result['y']=depthData.spatialCoordinates.y
                    result['z']=depthData.spatialCoordinates.z
                self.displayFaceQueue.put(result)
            
                

           

    def runObjectPipeline(self):
        firstTime = time.monotonic()
        startTime = time.monotonic()
        counter = 0
        fps = 0
        color = (255, 255, 255)
        colorFace = (255, 0, 0)
        
        text = TextHelper()
        facerec = FaceRecognition(self.databases, self.args.name)
        lastPersonAnnouncement = 0
        lastObjectAnnouncement = 0

        while True:
            # give other threads opportunity to run
            time.sleep(0.001)
            inDet = self.detectionNNQueue.tryGet()
           
            if inDet is not None:
              detections = inDet.detections
            
              self.displayYoloQueue.put(detections)
            
              if len(detections)>0 and time.time() - lastObjectAnnouncement > 10*60:
                  for detection in detections:    
                      print(detection.label)
                      try:
                          label = self.labelMap[detection.label]
                      except:
                          label = detection.label
                      if label != "person":
                          self.speakPub.publish("Detected %s" % label)
                  lastObjectAnnouncement = time.time()
    
    


    def runDisplay(self):
      
      color = (255, 255, 255)
      colorFace = (255, 0, 0)
      width = 1280
      height = 720
      text = TextHelper()
      frameSize = 720
      
      while True:
        time.sleep(0.001)
        #inPreview = self.previewQueue.get()
        #frame = inPreview.getCvFrame()
        inDepth = self.depthQueue.get()
        frameNotColor = inDepth.getFrame()
        frame = cv2.normalize(frameNotColor, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        frame = cv2.equalizeHist(frame)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
        #roi = roi.denormalize(width=frame.shape[1], height=frame.shape[0])
        width=frame.shape[1]
        height=frame.shape[0]
        #print(f"{frame.shape[1]}x{frame.shape[0]}")
        # If the frame is available, draw bounding boxes on it and show the frame
        

        if not self.displayYoloQueue.empty():
          detections = self.displayYoloQueue.get()
          for detection in detections:
              # Denormalize bounding box
              
              detection.xmax = (detection.xmax*frameSize+(1280-frameSize)/2)/1280
              detection.ymax = (detection.ymax*frameSize+(720-frameSize)/2)/720
              detection.xmin = (detection.xmin*frameSize+(1280-frameSize)/2)/1280
              detection.ymin = (detection.ymin*frameSize+(720-frameSize)/2)/720
            
              x1 = int(detection.xmin * width)
              x2 = int(detection.xmax * width)
              y1 = int(detection.ymin * height)
              y2 = int(detection.ymax * height)
              try:
                  label = self.labelMap[detection.label]
              except:
                  label = detection.label
              cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
              cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
              cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
              cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
              cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
              cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
              #print(f"{label} at {(detection.spatialCoordinates.x/1000):.2f}, {(detection.spatialCoordinates.y/1000):.2f}, {(detection.spatialCoordinates.z/1000):.2f}")
        
        if not self.displayFaceQueue.empty():
          result = self.displayFaceQueue.get()
          x1 = result['coords'][0]
          y1 = result['coords'][1]
          points = result['points']
          text.drawContours(frame, points)
          text.putText(frame, f"{result['name']} {(100*result['conf']):.0f}%", (x1,y1))
          cv2.putText(frame, f"X: {(result['x']):2f} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
          cv2.putText(frame, f"Y: {(result['y']):2f} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
          cv2.putText(frame, f"Z: {(result['z']):2f} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

        '''
        cfg = self.faceSpatialConfigQueue.tryGet()
        if cfg is not None:
          data = cfg.getConfigData()
          for datum in data:
            roi = datum.roi
            roi = roi.denormalize(1280, 720)
            x2 = int(roi.bottomRight().x) #*width)
            y2 = int(roi.bottomRight().y)#*height)
            x1 = int(roi.topLeft().x)#*width)
            y1 = int(roi.topLeft().y)#*height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)
        '''
        #cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)

        cv2.imshow("rgb2", frame)
        if cv2.waitKey(1) == ord('q'):
           break
