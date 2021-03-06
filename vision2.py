# coding=utf-8

'''
This demo consists of a pipeline that performs face recognition on the camera input with little
host side processing (almost all on device using script node)

The pipeline, in general, does the following (not all nodes are described):
1) In ImageManip node resizes the camera preview output and feeds them to a face detection NN.
1) A face detection NN generates an ImgDetections from frames and sends them to the script node.
2) The ImgDections contains a list of all faces detected.  For each detection, which consists
   of a bounding box, the script node creates an ImageManipConfig to crop out the faces,
   resize them, and change the output format.  This is sent to an ImageManip node. Note: one
   configuration per face is sent (and there may be more than one face detected per image).
3) An ImageManip node performs the crop/resize/format change and sends each face to a headpose
   estimator NN.
4) The headpose estimator NN calculates the headpose and sends them to the script node.
5) The script node reads (only) the "rotation" value from the headpose estimate and calculates
   a rotated rectangle for cropping that will work better with the face recognition node.
   The script creates an ImangeManipConfig with this rotated rectangle and sends to an
   ImageManip node.
6) An ImageManip node performs the crop and resize and sends each face to a face recognition NN.
4) The face recognition NN generates features based upon the face image sent.

The pipeline outputs (xlinks) the camera preview output, the face detections, the rotated
rectangle values, and the face recognition features.

Its important to note that there is a 1:N relationship between face detections and the latter
outputs of the pipeline.  For each face detection (i.e. ImgDetection) with N number of faces
detected, there are N number of rotated rectangle values and face recognition features.

This pipeline works because all node inputs are blocking and therefore, remain can remain in sync.
However, if the host does not process all of the output queues fast enough, its possible that
a node's output queue overruns and something will be dropped.  When this happens, the pipeline
can become out of sync and recognitions may not line up with their rotated rectangles.

This demo was tested on an OAK-D with a 5 FPS and two people in the database.  At a FPS of 10, there
were errors in the recognition and its suspected that one output queue might have overflowed.

An option is to create threads where one thread reads the output from the pipeline and stores it in
a Queue.  The second thread reads this Queue and outputs to the display.  This way you may miss
a recognition that that queue overflows because the frame rate is fast, but you don't overrun the
output queue of a node and lose sync (if this is what is really happening).
#

'''
import os
import json
from datetime import timedelta, datetime
import argparse
import blobconverter
import cv2
import depthai as dai
import numpy as np
import time
from queue import Queue
import threading
import pyray as ray
import math
from collections import Counter
#from ctypes import *

print(dai.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")

args = parser.parse_args()

DISPLAY_FACE = True
VIDEO_SIZE = (1072, 1072)
PREVIEW = False
OBJECT = True

databases = "databases"
if not os.path.exists(databases):
    os.mkdir(databases)

# Tiny yolo v3/4 label texts
labelMap = [
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

class HostSync:
    def __init__(self):
        self.array = []
    def add_msg(self, msg):
        self.array.append(msg)
    def get_msg(self, timestamp):
        def getDiff(msg, timestamp):
            return abs(msg.getTimestamp() - timestamp)
        if len(self.array) == 0: return None

        self.array.sort(key=lambda msg: getDiff(msg, timestamp))

        # Remove all frames that are older than 0.5 sec
        for i in range(len(self.array)):
            j = len(self.array) - 1 - i
            if getDiff(self.array[j], timestamp) > timedelta(milliseconds=500):
                self.array.remove(self.array[j])

        if len(self.array) == 0: return None
        return self.array.pop(0)

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
        self.name = name
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
        self.printed = True

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
        name = conf[0] if conf[0][0] >= 0.5 else (1 - conf[0][0], "UNKNOWN")
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
            with np.load(f"{databases}/{self.name}.npz") as db:
                db_ = [db[j] for j in db.files][:]
        except Exception as e:
            db_ = []
        db_.append(np.array(results))
        np.savez_compressed(f"{databases}/{self.name}", *db_)
        self.adding_new = False






class VisionSystem:

    def __init__(self):
        self.openvinoVersion = "2021.4"
        print(dai.__version__)
        parser = argparse.ArgumentParser()
        parser.add_argument("-name", "--name", type=str, help="Name of the person for database saving")
        self.args = parser.parse_args()
        self.packageDir = "."
        self.depthSync = False #start off out of sync
        
        self.databases = self.packageDir+"/databases"
        print(self.databases)
        if not os.path.exists(self.databases):
            os.mkdir(self.databases)



        self.pipeline = self.createPipeline()

        self.device = dai.Device(self.pipeline)
        cameraInfo = self.device.getCameraSensorNames()
        cameraModel = cameraInfo[dai.CameraBoardSocket.RGB]
        if cameraModel == "IMX214":
            self.FPS = 30
            self.frameSkip = 4
        else:
            self.FPS = 30
            self.frameSkip = 4

        if PREVIEW:
            self.frameQ = self.device.getOutputQueue("frame", 15, False)
        self.recCfgQ = self.device.getOutputQueue("face_rec_cfg_out", 15, False)
        self.arcQ = self.device.getOutputQueue("arc_out", 15, False)
        self.detQ = self.device.getOutputQueue("faceDetections", 15, False)
        self.headposeQ = self.device.getOutputQueue("headpose_out", 15, False)
        self.cameraControlQ = self.device.getInputQueue("cameraControl",4,False)
        

        self.recognizeFaceQueue = Queue(maxsize = 10)

        self.faceThread = threading.Thread(target=self.runFaceDetectThread)
        self.faceThread.daemon = True
        self.faceThread.start()

        self.recognizeThread = threading.Thread(target=self.runRecognizeFaceThread)
        self.recognizeThread.daemon = True
        self.recognizeThread.start()

        if OBJECT:
            self.objectQ = self.device.getOutputQueue(name="objectDetections", maxSize=15, blocking=False)
            self.depthSyncQ = self.device.getOutputQueue(name="depthSyncOut", maxSize=15, blocking=False)
            self.videoDisplayObjectQueue = Queue(maxsize = 20)
            self.trackedObjectManager = TrackedObjectManager()
            self.objectThread = threading.Thread(target=self.runObjectThread)
            self.objectThread.daemon = True
            self.objectThread.start()


        if PREVIEW:
            print("Running display thread.")
            self.videoDisplayFaceQueue = Queue(maxsize = 20)
            self.videoDisplayThread = threading.Thread(target=self.runVideoDisplayThread)
            self.videoDisplayThread.daemon = True
            self.videoDisplayThread.start()

        if DISPLAY_FACE:
            self.faceQ = self.device.getOutputQueue("face", 15, False)
            self.raylibDisplayFaceQueue = Queue(maxsize = 20)
            self.raylibDisplayThread = threading.Thread(target=self.runRaylibDisplayThread)
            self.raylibDisplayThread.daemon = True
            self.raylibDisplayThread.start()


    def runFaceDetectThread(self):
        results = {}
        while True:
            time.sleep(0.001)
            inDet = self.detQ.tryGet()
            if inDet is not None:
                for det in inDet.detections:
                    cfg = self.recCfgQ.get()
                    arcIn = self.arcQ.get()
                    headpose = self.headposeQ.get()
                    if DISPLAY_FACE:
                        face = self.faceQ.get()
                    rr = cfg.getRaw().cropConfig.cropRotatedRect
                    h, w = VIDEO_SIZE
                    center = (int(rr.center.x * w), int(rr.center.y * h))
                    size = (int(rr.size.width * w), int(rr.size.height * h))
                    rotatedRect = (center, size, rr.angle)
                    points = np.int0(cv2.boxPoints(rotatedRect))
                    features = np.array(arcIn.getFirstLayerFp16())
                    values = self.to_tensor_result(headpose)
                    result = {
                        'name': "",
                        'conf': "",
                        'coords': center,
                        'points': points,
                        'ts': time.time(),
                        'xmin': det.xmin,
                        'ymin': det.ymin,
                        'xmax': det.xmax,
                        'ymax': det.ymax,
                        'x': det.spatialCoordinates.x,
                        'y': det.spatialCoordinates.y,
                        'z': det.spatialCoordinates.z,
                        'features': features,
                        'yaw': values['angle_y_fc'][0][0],
                        'pitch': values['angle_p_fc'][0][0],
                        'roll': values['angle_r_fc'][0][0],                        
                    }
                    if DISPLAY_FACE:
                        result['face']=face
                    if self.recognizeFaceQueue.full():
                        self.recognizeFaceQueue.get()
                    self.recognizeFaceQueue.put(result)

    def to_tensor_result(self, packet):
        return {
            tensor.name: np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
            for tensor in packet.getRaw().tensors
        }
    
    
    
    def runRecognizeFaceThread(self):
        facerec = FaceRecognition(self.databases, self.args.name)
        # once face recognition is up and running, turn on the frame processing
        print("Turning on processing")
        dict = {'frameSkip':self.frameSkip}
        data = json.dumps(dict).encode('utf-8')
        buffer = dai.Buffer()
        buffer.setData(list(data))
        self.cameraControlQ.send(buffer)

        while True:
            time.sleep(0.001)
            if not self.recognizeFaceQueue.empty():
                result = self.recognizeFaceQueue.get()
                conf, name = facerec.new_recognition(result['features'])
                result['name'] = name
                result['conf'] = conf
                #print(f"{result['name']} at {(100*result['conf']):.0f}%, {result['x']:.2f}, {result['y']:.2f}, {result['z']:.2f},")
                if PREVIEW:
                    self.videoDisplayFaceQueue.put(result)
                if DISPLAY_FACE:
                    self.raylibDisplayFaceQueue.put(result)
                if self.depthSync == 1:  #we are using the sync message in the runObjectThread to make sure depth/stereo frames are in sync. 
                    self.trackedObjectManager.processFaceDetection(result, time.time())  # done this way to easier know if multiple items exist
                
    def runObjectThread(self):
        time.sleep(0.001)
        while True:
            inDet = self.objectQ.tryGet()
            if inDet is not None:
                detections = inDet.detections
                data = self.depthSyncQ.get().getData()
                jsonstr = str(data, 'utf-8')
                self.depthSync = json.loads(jsonstr)
                if PREVIEW:
                    self.videoDisplayObjectQueue.put(detections)
                if self.depthSync == 1 and len(detections)>0:
                    for detection in detections:
                        try:
                            label = labelMap[detection.label]
                        except:
                            label = detection.label
                    self.trackedObjectManager.processObjectDetections(detections, time.time())  # done this way to easier know if multiple items exist
                        #print(f"{label} at {detection.spatialCoordinates.x:.2f}, {detection.spatialCoordinates.y:.2f}, {detection.spatialCoordinates.z:.2f}")

    def runRaylibDisplayThread(self):
        text = TextHelper()

        ray.init_window(1024, 600, "Hello Raylib")
        ray.toggle_fullscreen()
        ray.hide_cursor()
        ray.set_target_fps(60)
        ray.set_trace_log_level(ray.LOG_ERROR)
        width = ray.get_screen_width()
        height = ray.get_screen_height()
        #width = 800
        #height = 450
        print(width)
        while True:
            time.sleep(0.001)
            while not ray.window_should_close():
                if not self.raylibDisplayFaceQueue.empty():
                    result=self.raylibDisplayFaceQueue.get()
                    frame = result['face'].getCvFrame()
                    #frame = self.draw_3d_axis(frame, int(frame.shape[1]/2), int(frame.shape[0]/2), result['yaw'], result['pitch'], result['roll'])
                    h0, w0 = frame.shape[:2]
                    frame = cv2.resize(frame, (w0*4, h0*4), interpolation=cv2.INTER_LANCZOS4)
                    frame = cv2.Canny(frame, 25, 100)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ffiData = ray.ffi.from_buffer(frame.data)
                    image = ray.Image(ffiData,frame.shape[1],frame.shape[0],1,ray.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
                    texture = ray.load_texture_from_image(image)
                    #ray.unload_image(image)
                    ray.begin_drawing()
                    ray.clear_background(ray.BLACK)
                    scale = width/frame.shape[0]
                    scale1 = height/frame.shape[1]
                    if scale>scale1:
                        scale = scale1
                    #vector = ray.Vector2(int(200-frame.shape[1]/2), 0)
                    vector = ray.Vector2(int((width-height)/2),height)
                    ray.draw_texture_ex(texture, vector, -90, scale, ray.RAYWHITE)
                    ray.end_drawing()
            ray.unload_texture(texture)
            ray.close_window()
    
    def draw_3d_axis(self, frame, origin_x, origin_y, y, p, r):
        size = 50
        yaw = y*3.141592/180
        pitch = p*3.141592/180
        roll = r*3.141592/180
        
        sinY = math.sin(yaw)
        sinP = math.sin(pitch)
        sinR = math.sin(roll)
        
        cosY = math.cos(yaw)
        cosP = math.cos(pitch)
        cosR = math.cos(roll)
        
        # X axis (red)
        x1 = origin_x + size * (cosR * cosY + sinY * sinP * sinR)
        y1 = origin_y + size * cosP * sinR
        cv2.line(frame, (origin_x, origin_y), (int(x1), int(y1)), (0, 0, 255), 1)

        # Y axis (green)
        x2 = origin_x + size * (cosR * sinY * sinP + cosY * sinR)
        y2 = origin_y - size * cosP * cosR
        cv2.line(frame, (origin_x, origin_y), (int(x2), int(y2)), (0, 255, 0), 1)

        # Z axis (blue)
        x3 = origin_x + size * (sinY * cosP)
        y3 = origin_y + size * sinP
        cv2.line(frame, (origin_x, origin_y), (int(x3), int(y3)), (255, 0, 0), 1)

        return frame


    def runVideoDisplayThread(self):
        text = TextHelper()
        frameDelay = 0
        frameArr = []
        frame = None
        while True:
            time.sleep(0.001)
            frameIn = self.frameQ.tryGet()
            if frameIn is not None:
                frameArr.append(frameIn.getCvFrame())
                # The original version of this demo delayed the video ouput to sync the detections
                # with the video (head pose estimation and face recognition takes about ~200ms).
                # However, the frame rate was slowed down to avoid what I suspect are drops in the
                # output queue (which I believe lead to sync issues).
                # At the current frame rate (5 fps), 0 frame delay appears to work well enough.
                if frameDelay < len(frameArr):
                    frame = frameArr.pop(0)
            if frame is not None:
                width = frame.shape[1]
                height = frame.shape[0]
                if not self.videoDisplayFaceQueue.empty():
                    result=self.videoDisplayFaceQueue.get()
                    text.drawContours(frame, result['points'])
                    text.putText(frame, f"{result['name']} {(100*result['conf']):.0f}%", result['coords'])
                    text.putText(frame, f"x:{result['x']:0.2f}", (result['coords'][0],result['coords'][1]+30))
                    text.putText(frame, f"y:{result['y']:0.2f}", (result['coords'][0],result['coords'][1]+60))
                    text.putText(frame, f"z:{result['z']:0.2f}", (result['coords'][0],result['coords'][1]+90))

                if OBJECT:
                    if not self.videoDisplayObjectQueue.empty():
                        detections = self.videoDisplayObjectQueue.get()
                        for detection in detections:
                            # Denormalize bounding box
                            x2 = int(detection.xmax*width)
                            y2 = int(detection.ymax*height)
                            x1 = int(detection.xmin*width)
                            y1 = int(detection.ymin*height)
                            try:
                                label = labelMap[detection.label]
                            except:
                                label = detection.label
                            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), cv2.FONT_HERSHEY_SIMPLEX)
                cv2.imshow("color", cv2.resize(frame, (800,800)))
                if cv2.waitKey(1) == ord('q'):
                    break



    def createPipeline(self):
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
        openvino_version = '2021.4'

        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera)
        # For ImageManip rotate you need input frame of multiple of 16
        cam.setPreviewSize(1072, 1072)
        cam.setVideoSize(VIDEO_SIZE)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setFps(30) #this will be 'paced' by the script by a factor of 6, so 5 FPS

        if PREVIEW:
            host_face_out = pipeline.create(dai.node.XLinkOut)
            host_face_out.setStreamName('frame')
            cam.video.link(host_face_out.input)

        # ImageManip that will crop the frame before sending it to the Face detection NN node
        face_det_manip = pipeline.create(dai.node.ImageManip)
        face_det_manip.initialConfig.setResize(300, 300)
        face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

        # NeuralNetwork
        print("Creating Face Detection Neural Network...")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        face_det_nn.setConfidenceThreshold(0.5)
        face_det_nn.setBlobPath(blobconverter.from_zoo(
            name="face-detection-retail-0004",
            shaves=6,
            version=openvino_version
        ))
        face_det_nn.setBoundingBoxScaleFactor(0.50)
        face_det_nn.setDepthLowerThreshold(10)
        face_det_nn.setDepthUpperThreshold(5000)
        face_det_nn.setConfidenceThreshold(0.3)
        face_det_nn.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.MIN)

        # Link Face ImageManip -> Face detection NN node
        face_det_manip.out.link(face_det_nn.input)

        # Script node will take the output from the face detection NN as an input and set ImageManipConfig
        # to the 'age_gender_manip' to crop the initial frame
        script = pipeline.create(dai.node.Script)
        script.setProcessor(dai.ProcessorType.LEON_CSS)

        face_det_nn.out.link(script.inputs['face_det_in'])

        face_detections_out = pipeline.create(dai.node.XLinkOut)
        face_detections_out.setStreamName('faceDetections')
        face_det_nn.out.link(face_detections_out.input)

        # We are only interested in timestamp, so we can sync depth frames with NN output
        face_det_nn.passthrough.link(script.inputs['face_pass'])

        with open("spatialScript.py", "r") as f:
            script.setScript(f.read())

        # create a camera control XLINKIN to script
        cameraControlxin = pipeline.create(dai.node.XLinkIn)
        cameraControlxin.setStreamName('cameraControl')
        cameraControlxin.out.link(script.inputs['cameraControl'])

        # ImageManip as a workaround to have more frames in the pool.
        # cam.preview can only have 4 frames in the pool before it will
        # wait (freeze). Copying frames and setting ImageManip pool size to
        # higher number will fix this issue.


        copy_manip = pipeline.create(dai.node.ImageManip)
        cam.preview.link(copy_manip.inputImage)
        copy_manip.setNumFramesPool(20)
        copy_manip.setMaxOutputFrameSize(1072*1072*3)

        #copy_manip.out.link(face_det_manip.inputImage)
        script.outputs['pacedPreview'].link(face_det_manip.inputImage)
        copy_manip.out.link(script.inputs['preview'])

        print("Creating Head pose estimation NN")
        headpose_manip = pipeline.create(dai.node.ImageManip)
        headpose_manip.setWaitForConfigInput(True) # needed to maintain sync
        headpose_manip.initialConfig.setResize(60, 60)

        script.outputs['manip_cfg'].link(headpose_manip.inputConfig)
        script.outputs['manip_img'].link(headpose_manip.inputImage)

        headpose_nn = pipeline.create(dai.node.NeuralNetwork)
        headpose_nn.setBlobPath(blobconverter.from_zoo(
            name="head-pose-estimation-adas-0001",
            shaves=6,
            version=openvino_version
        ))
        headpose_manip.out.link(headpose_nn.input)
        
        if DISPLAY_FACE:
            xout_face = pipeline.createXLinkOut()
            xout_face.setStreamName('face')
            headpose_manip.out.link(xout_face.input)

        headpose_nn.out.link(script.inputs['headpose_in'])
        headpose_nn.passthrough.link(script.inputs['headpose_pass'])
        
        headpose_out = pipeline.create(dai.node.XLinkOut)
        headpose_out.setStreamName('headpose_out')
        headpose_nn.out.link(headpose_out.input)

        print("Creating face recognition ImageManip/NN")

        face_rec_manip = pipeline.create(dai.node.ImageManip)
        face_rec_manip.setWaitForConfigInput(True) # needed to maintain sync
        face_rec_manip.initialConfig.setResize(112, 112)
        #face_rec_manip.initialConfig.setResize(128, 128)

        script.outputs['manip2_cfg'].link(face_rec_manip.inputConfig)
        script.outputs['manip2_img'].link(face_rec_manip.inputImage)

        face_rec_cfg_out = pipeline.create(dai.node.XLinkOut)
        face_rec_cfg_out.setStreamName('face_rec_cfg_out')
        script.outputs['manip2_cfg'].link(face_rec_cfg_out.input)

        # Only send metadata for the host-side sync
        # pass2_out = pipeline.create(dai.node.XLinkOut)
        # pass2_out.setStreamName('pass2')
        # pass2_out.setMetadataOnly(True)
        # script.outputs['manip2_img'].link(pass2_out.input)

        face_rec_nn = pipeline.create(dai.node.NeuralNetwork)
        # Removed from OMZ, so we can't use blobconverter for downloading, see here:
        # https://github.com/openvinotoolkit/open_model_zoo/issues/2448#issuecomment-851435301
        #face_rec_nn.setBlobPath("models/face-recognition-mobilefacenet-arcface_2021.2_4shave.blob")
        face_rec_nn.setBlobPath(str(blobconverter.from_zoo(
            name="face-reidentification-retail-0095",
            shaves=6,
            version=self.openvinoVersion
        )))

        face_rec_manip.out.link(face_rec_nn.input)

        arc_out = pipeline.create(dai.node.XLinkOut)
        arc_out.setStreamName('arc_out')
        face_rec_nn.out.link(arc_out.input)

        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)

        stereo.initialConfig.setConfidenceThreshold(255)

        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        stereo.depth.link(face_det_nn.inputDepth)

        if OBJECT:
            # Create object detection NN
            print("Creating Object Detection Neural Network...")
            yoloDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
            yoloDetectionNetwork.setBlobPath(str(blobconverter.from_zoo(
                name="yolov4_tiny_coco_416x416",
                zoo_type="depthai",
                shaves=6,
            )))

            yoloDetectionNetwork.setConfidenceThreshold(0.5)
            yoloDetectionNetwork.input.setBlocking(False)
            yoloDetectionNetwork.setBoundingBoxScaleFactor(0.25)
            yoloDetectionNetwork.setDepthLowerThreshold(100)
            yoloDetectionNetwork.setDepthUpperThreshold(5000)
            yoloDetectionNetwork.setSpatialCalculationAlgorithm(dai.SpatialLocationCalculatorAlgorithm.MIN)

            # Yolo specific parameters
            yoloDetectionNetwork.setNumClasses(80)
            yoloDetectionNetwork.setCoordinateSize(4)
            yoloDetectionNetwork.setAnchors(np.array([23,27, 37,58, 81,82, 81,82, 135,169, 344,319]))
            yoloDetectionNetwork.setAnchorMasks({ "side26": np.array([0,1,2]), "side13": np.array([3,4,5]) })
            yoloDetectionNetwork.setIouThreshold(0.35)

            object_det_manip = pipeline.create(dai.node.ImageManip)
            object_det_manip.initialConfig.setResize(416, 416)
            object_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
            script.outputs['pacedPreview'].link(object_det_manip.inputImage)
            object_det_manip.out.link(yoloDetectionNetwork.input)
            stereo.depth.link(yoloDetectionNetwork.inputDepth)

            xoutNN = pipeline.create(dai.node.XLinkOut)
            xoutNN.setStreamName("objectDetections")
            yoloDetectionNetwork.out.link(xoutNN.input)
            
            yoloDetectionNetwork.passthrough.link(script.inputs['yoloPass'])
            yoloDetectionNetwork.passthroughDepth.link(script.inputs['yoloPassDepth'])

            depthSync_out = pipeline.create(dai.node.XLinkOut)
            depthSync_out.setStreamName('depthSyncOut')
            script.outputs['depthSync'].link(depthSync_out.input)
            
            
        return pipeline

class TrackedObjectManager:
    '''
    This class is intended to manage the tracked objects, determine which
    detection belongs to which object, and attempt to allow for multiple objects
    of the same label to be tracked.  It will accomplish this by tracking their locations
    and determine which detection belongs to which object.
    '''

    
    def __init__(self):
        print("initializing tracked object manager")
        self.trackedObjects = []
        self.trackedObjectsTracker = {}
        self.objectLocationThreshold = [
            40,               40,           200,              100,            10000,         5000,            6000,
            300,              400,         100,              300,            500,           200,             50,
            20,               20,           40,               100,            40,            100,             200,
            100,             100,           200,              20,             20,            20,              20,
            30,               20,           20,               20,             20,            100,             20,
            20,               20,           50,               20,             20,            10,              10,
            5,                5,             5,               10,             10,            10,              10,
            5,                5,             5,                5,             20,            5,               20,
            20,               100,          50,               50,             100,           200,             10,
            10,               20,           20,               20,             40,            100,             50,
            50,               50,           100,              5,              100,           20,              20,
            50,               50,           10,
        ]

    def processObjectDetections(self, detections, time):
        #determine how many of each label exists in frame and if quantity is less than what's tracked, create a new one
        labels = []
        for detection in detections:
            labels.append(detection.label)
        objects = Counter(labels)
        print(objects)
        for object in objects:
            numberOfObjectsToAdd = 0
            if object in self.trackedObjectsTracker:
                count = self.trackedObjectsTracker[object]["count"]
                if count < objects[object]:
                    numberOfObjectsToAdd = objects[object]-count
            else:
                numberOfObjectsToAdd = objects[object]
            if numberOfObjectsToAdd>0:
                for x in range(numberOfObjectsToAdd):
                    if object == 0: #person
                        newObject = TrackedPerson(object, None, 0)
                    else:
                        newObject = TrackedObject(object, None, 0)
                    self.trackedObjects.append(newObject)
                    self.addNewTrackedObjectToTracker(newObject)
                    print(f"new object {newObject.name} tracked")
        
        for detection in detections:
            candidate = None
            minDistance = 99999
            for object in self.trackedObjects:
                if object.label == detection.label:
                    #if object.label == 0:
                    if False: #I think we should always track person coordinates to person coordinates.
                        distanceToCurrentFace = self.computeDistanceOfDetectionFromObject(object.currentFaceLocation, detection.spatialCoordinates, time )
                        distanceToPreviousFace = self.computeDistanceOfDetectionFromObject(object.previousFaceLocation, detection.spatialCoordinates, time )
                        distance = distanceToCurrentFace if distanceToCurrentFace<distanceToPreviousFace else distanceToPreviousFace
                    else:
                        distanceToCurrent = self.computeDistanceOfDetectionFromObject(object.currentLocation, detection.spatialCoordinates, time )
                        distanceToPrevious = self.computeDistanceOfDetectionFromObject(object.previousLocation, detection.spatialCoordinates, time ) if distanceToCurrent<=6000 else 7000
                        distance = distanceToCurrent if distanceToCurrent<distanceToPrevious else distanceToPrevious
                    if distance < minDistance:
                        minDistance = distance
                        candidate = object
            candidate.updateLocation(detection, time)
            #print(f"{candidate.getName()} position updated, ({detection.spatialCoordinates.x:0.2f}, {detection.spatialCoordinates.y:0.2f}, {detection.spatialCoordinates.z:0.2f}) distance = {minDistance}.")
        self.reportTrackedObjects()

    def processFaceDetection(self, result, time):
        #determine how many of each label exists in frame and if quantity is less than what's tracked, create a new one
        if result["name"] == "UNKNOWN":
            return #not sure what to do here.. if an unknown person lasts long enough, then need to process through adding them to the face recognition database
        candidate = None
        maxArea = 0
        spatial = dai.Point3f(result["x"], result["y"], result["z"])
        bb = (result["xmin"], result["ymin"], result["xmax"], result["ymax"])
        for object in self.trackedObjects: #find best match.
            if object.name == "person":
                #distanceToCurrent = self.computeDistanceOfDetectionFromObject(object.currentLocation, spatial, time )
                #distanceToPrevious = self.computeDistanceOfDetectionFromObject(object.previousLocation, spatial, time ) if distanceToCurrent<=6000 else 7000
                overlapOfCurrent = self.computeOverlap(object.currentBoundingBox, bb, time)
                overlapOfPrevious = self.computeOverlap(object.previousBoundingBox, bb, time)
                area = overlapOfCurrent if overlapOfCurrent>overlapOfPrevious else overlapOfPrevious
                if area > maxArea:
                    maxArea = area
                    candidate = object
        if candidate is not None:
            candidate.updateName(result["name"])
            candidate.updateFaceLocation(spatial, time)
            

    def reportTrackedObjects(self):
        for object in self.trackedObjects:
            if object.label == 0: #person
                print(f"{object.name} at {object.currentLocation[0]:.2f}, {object.currentLocation[1]:.2f}, {object.currentLocation[2]:.2f} face at {object.currentFaceLocation[0]:.2f}, {object.currentFaceLocation[1]:.2f}, {object.currentFaceLocation[2]:.2f}")
            else:
                print(f"{object.name} at {object.currentLocation[0]:.2f}, {object.currentLocation[1]:.2f}, {object.currentLocation[2]:.2f}")
        print("#--------#")
                
    def computeDistanceOfDetectionFromObject(self, l, c, t):
        if l[3]>0:
            if t-l[3]>0.01: # if its 0 (means new object) or if the timestamp has not been updated (meaning its already been matched), do this
                return math.sqrt( (l[0]-c.x)**2 + (l[1]-c.y)**2 + (l[2]-c.z)**2 ) /100 #/ (t-l[3]) / 100 #scaling factor
            else:
                return 7000
        else:
            return 6000 #return something far away.. 

    def computeOverlap(self, l, c, t):
        dx = min(l[2], c[2]) - max(l[0], c[0])
        dy = min(l[3], c[3]) - max(l[1], c[1])
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            return 0
        

    def addNewTrackedObjectToTracker(self, newObject):
        if newObject.label in self.trackedObjectsTracker:
            self.trackedObjectsTracker[newObject.label]["count"]=self.trackedObjectsTracker[newObject.label]["count"]+1
            return
        self.trackedObjectsTracker[newObject.label]={"name":newObject.name, "count":1}
                
            

class TrackedObject:

    def __init__(self):
        self.ID = 0
        self.label = 0
        self.name = 0
        self.previousLocation = (0, 0, 0, 0)
        self.currentLocation = (0, 0, 0, 0)
        self.futureLocation = (0, 0, 0, 0)
        self.previousBoundingBox = (0, 0, 0, 0, 0)
        self.currentBoundingBox = (0, 0, 0, 0, 0)

    def __init__(self, label, c, t):
        self.ID = 0
        self.label = label
        try:
            self.name = labelMap[label]
        except:
            self.name = label
        if c is not None:
            self.previousLocation = (c.spatialCoordinates.x, c.spatialCoordinates.y, c.spatialCoordinates.z, t)
            self.currentLocation = (c.spatialCoordinates.x, c.spatialCoordinates.y, c.spatialCoordinates.z,t )
            self.futureLocation = (c.spatialCoordinates.x, c.spatialCoordinates.y, c.spatialCoordinates.z,t )
            self.previousBoundingBox = (c.xmin, c.ymin, c.xmax, c.ymax, t)
            self.currentBoundingBox = (c.xmin, c.ymin, c.xmax, c.ymax, t)
        else:
            self.previousLocation = (0, 0, 0, t)
            self.currentLocation = (0, 0, 0, t)
            self.futureLocation = (0, 0, 0, t)
            self.previousBoundingBox = (0, 0, 0, 0, t)
            self.currentBoundingBox = (0, 0, 0, 0, t)
                
    def updateName(self, name):
        self.name = name
        
    def updateLocation(self, c, t):
        #todo: add one euro filter or kalman filter
        self.previousLocation = (self.currentLocation[0], self.currentLocation[1], self.currentLocation[2], self.currentLocation[3])
        self.currentLocation = (c.spatialCoordinates.x, c.spatialCoordinates.y, c.spatialCoordinates.z, t)
        self.previousBoundingBox = (self.currentBoundingBox[0], self.currentBoundingBox[1], self.currentBoundingBox[2], self.currentBoundingBox[3], self.currentBoundingBox[4])
        self.currentBoundingBox = (c.xmin, c.ymin, c.xmax, c.ymax, t)

    def getName(self):
        return self.name
        
class TrackedPerson(TrackedObject):
    def __init__(self, label, c, t):
        self.currentFaceLocation = (0, 0, 0, 0)
        self.previousFaceLocation = (0, 0, 0, 0)
        TrackedObject.__init__(self, label, c, t)
    
    def updateFaceLocation(self, c, t):
        self.previousFaceLocation = (self.currentFaceLocation[0], self.currentFaceLocation[1], self.currentFaceLocation[2], self.currentFaceLocation[3])
        self.currentFaceLocation = (c.x, c.y, c.z, t)

if __name__ == '__main__':
    print("instantiating vision system.")
    _vision=VisionSystem()
    while True:
      time.sleep(0.001)
      if False:
         break
