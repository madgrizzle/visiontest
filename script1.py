import time
import json

bboxes = [] # List of face BBs
l = [] # List of images

# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def remove_prev_frames(seq):
    if len(l) == 0:
        return
    for rm, frame in enumerate(l):
        if frame.getSequenceNum() == seq:
            # node.warn(f"List len {len(l)} Frame with same seq num: {rm},seq {seq}")
            break
    for i in range(rm):
        l.pop(0)

def find_frame(seq):
    for frame in l:
        if frame.getSequenceNum() == seq:
            return frame
def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.0
    if bb.ymin < 0: bb.ymin = 0.0
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb
    
faceROIScale = 0.5 #0.0625

#faceROIxOffset = (640-416)/640/2
#faceROIyOffset = (400-416)/400/2
while True:
    time.sleep(0.001)
    preview = node.io['preview'].tryGet()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}, size {len(l)}")
        l.append(preview)
        # Max pool size is 10.
        if 10 < len(l):
            l.pop(0)

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        # node.warn(f"New detection start")
        passthrough = node.io['face_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue

        img = find_frame(seq) # Matching frame is the first in the list
        if img is None:
            continue

        for det in face_dets.detections:
            bboxes.append(det) # For the rotation
            cfg = ImageManipConfig()
            correct_bb(det)
            BB =  {
                    'xmax': det.xmax,
                    'ymax': det.ymax,
                    'xmin': det.xmin,
                    'ymin': det.ymin,
                }
            b=Buffer(80)
            b.setData(json.dumps(BB).encode('utf-8'))

            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(60, 60)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)
            spatialConfigData = SpatialLocationCalculatorConfigData()
            spatialConfigData.depthThresholds.lowerThreshold = 100
            spatialConfigData.depthThresholds.upperThreshold = 10000
            
            frameSize = 720

            xmax = (det.xmax*frameSize+(1280-frameSize)/2)/1280
            ymax = (det.ymax*frameSize+(720-frameSize)/2)/720
            xmin = (det.xmin*frameSize+(1280-frameSize)/2)/1280
            ymin = (det.ymin*frameSize+(720-frameSize)/2)/720

            centerx = (xmax + xmin)/2.0
            centery = (ymax + ymin)/2.0
            #node.warn(f"0:{det.xmax},{det.ymax} {det.xmin},{det.ymin}")
            #node.warn(f"0:{det.xmax-det.xmin},{det.ymax-det.ymin}")
            xmax = (xmax-centerx)*faceROIScale+centerx
            ymax = (ymax-centery)*faceROIScale+centery
            xmin = (xmin-centerx)*faceROIScale+centerx
            ymin = (ymin-centery)*faceROIScale+centery
            if xmin < 0: xmin = 0.0
            if ymin < 0: ymin = 0.0
            if xmax > 1: xmax = 0.999
            if ymax > 1: ymax = 0.999
            #node.warn(f"1:{det.xmax},{det.ymax} {det.xmin},{det.ymin}")
            #node.warn(f"1:{det.xmax-det.xmin},{det.ymax-det.ymin}")
            #node.warn(f"1:{centerx},{centery}")
            bottomRight = Point2f(xmax, ymax)
            topLeft = Point2f(xmin, ymin)
            spatialConfigData.roi = Rect(topLeft, bottomRight)
            
            spatialConfigData.calculationAlgorithm = SpatialLocationCalculatorAlgorithm.AVERAGE
            spatialConfig = SpatialLocationCalculatorConfig()
            spatialConfig.addROI(spatialConfigData)
            node.io['spatialFaceConfig'].send(spatialConfig)
            node.io['spatialFaceConfigBB'].send(b)
          
    headpose = node.io['headpose_in'].tryGet()
    if headpose is not None:
        # node.warn(f"New headpose")
        passthrough = node.io['headpose_pass'].get()
        seq = passthrough.getSequenceNum()
        # node.warn(f"New headpose seq {seq}")
        # Face rotation in degrees
        r = headpose.getLayerFp16('angle_r_fc')[0] # Only 1 float in there
        bb = bboxes.pop(0) # Get BB from the img detection
        correct_bb(bb)

        # remove_prev_frame(seq)
        remove_prev_frames(seq)
        if len(l) == 0:
            continue
        img = l.pop(0) # Matching frame is the first in the list
        # node.warn('HP' + str(img))
        # node.warn('bb' + str(bb))
        cfg = ImageManipConfig()
        rr = RotatedRect()
        rr.center.x = (bb.xmin + bb.xmax) / 2
        rr.center.y = (bb.ymin + bb.ymax) / 2
        rr.size.width = bb.xmax - bb.xmin
        rr.size.height = bb.ymax - bb.ymin
        rr.angle = r # Rotate the rect in opposite direction
        # True = coordinates are normalized (0..1)
        cfg.setCropRotatedRect(rr, True)
        #cfg.setResize(128, 128)
        cfg.setResize(96,112)
        cfg.setKeepAspectRatio(True)

        node.io['manip2_cfg'].send(cfg)
        node.io['manip2_img'].send(img)
        
        BB =  {
                    'xmax': bb.xmax,
                    'ymax': bb.ymax,
                    'xmin': bb.xmin,
                    'ymin': bb.ymin,
                }
        b=Buffer(80)
        b.setData(json.dumps(BB).encode('utf-8'))
        node.io['face_recBB'].send(b)
