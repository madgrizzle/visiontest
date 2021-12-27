import time
import json
bboxes = [] # List of face BBs
l = [] # List of images

# So the correct frame will be the first in the list
# For this experiment this function is redundant, since everything
# runs in blocking mode, so no frames will get lost
def find_frame(seq):
    for frame in l:
        if frame.getSequenceNum() == seq:
            return(frame)
            break
    return None

def correct_bb(bb):
    if bb.xmin < 0: bb.xmin = 0.0
    if bb.ymin < 0: bb.ymin = 0.0
    if bb.xmax > 1: bb.xmax = 0.999
    if bb.ymax > 1: bb.ymax = 0.999
    return bb

frameCount = -1
while True:
    time.sleep(0.001)
    preview = node.io['preview'].tryGet()
    if preview is not None:
        # node.warn(f"New frame {preview.getSequenceNum()}, size {len(l)}")
        if frameCount != -1:  #don't process frames until cameracontrol message is received
            frameCount = frameCount + 1
            if frameCount == frameSkip:
                frameCount = 0
                l.append(preview)
                node.io['pacedPreview'].send(preview)
                # Max pool size is 18.
                if 18 < len(l):
                    l.pop(0)
    
    cameraControl = node.io['cameraControl'].tryGet()
    if cameraControl is not None:
        data = cameraControl.getData()
        jsonstr = str(data, 'utf-8')
        dict = json.loads(jsonstr)
        frameSkip = dict['frameSkip']
        frameCount = 0 #this turns on processing of the preview frames

    face_dets = node.io['face_det_in'].tryGet()
    if face_dets is not None:
        # node.warn(f"New detection start")
        passthrough = node.io['face_pass'].get()
        seq = passthrough.getSequenceNum()
        #node.warn(f"New detection {seq}")
        if len(l) == 0:
            continue

        img = find_frame(seq) # Matching frame is the first in the list
        if img is None:
            node.warn(f"Frame not found")
            continue

        for det in face_dets.detections:
            bboxes.append(det) # For the rotation
            cfg = ImageManipConfig()
            correct_bb(det)
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(60, 60)
            cfg.setKeepAspectRatio(False)
            node.io['manip_cfg'].send(cfg)
            node.io['manip_img'].send(img)

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
        img = find_frame(seq)
        if img is None:
          node.warn(f"Frame not found.  Sync problem will occur")
          continue
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
        #cfg.setResize(112, 112)
        cfg.setResize(128, 128)
        cfg.setKeepAspectRatio(True)

        node.io['manip2_cfg'].send(cfg)
        node.io['manip2_img'].send(img)