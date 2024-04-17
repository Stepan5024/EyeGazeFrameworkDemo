import cv2
import numpy as np
from .results import GazeResultContainer

def draw_gaze(a,b,c,d,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = c
    pos = (int(a+c / 2.0), int(b+d / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[1])
    tx = -int(w/2) * np.sin(pitchyaw[0]) * np.cos(pitchyaw[1]) + int(w/2)
    ty = -int(h/2) * np.sin(pitchyaw[1]) + int(h/2)
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    cv2.circle(image_out, (int(tx), int(ty)), 10, color=(0, 0, 255))
    return image_out, tx, ty, h, w

def draw_bbox(frame: np.ndarray, bbox: np.ndarray):
    
    x_min=int(bbox[0])
    if x_min < 0:
        x_min = 0
    y_min=int(bbox[1])
    if y_min < 0:
        y_min = 0
    x_max=int(bbox[2])
    y_max=int(bbox[3])

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    return frame

def render(frame: np.ndarray, results: GazeResultContainer):

    # Draw bounding boxes
    for bbox in results.bboxes:
        frame = draw_bbox(frame, bbox)

    # Draw Gaze
    for i in range(results.pitch.shape[0]):

        bbox = results.bboxes[i]
        pitch = results.pitch[i]
        yaw = results.yaw[i]
        
        # Extract safe min and max of x,y
        x_min=int(bbox[0])
        if x_min < 0:
            x_min = 0
        y_min=int(bbox[1])
        if y_min < 0:
            y_min = 0
        x_max=int(bbox[2])
        y_max=int(bbox[3])

        # Compute sizes
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))
        _, x, y, h, w = draw_gaze(x_min,y_min,bbox_width, bbox_height,frame,(pitch,yaw),color=(0,0,255))
        cv2.circle(frame, (int(x), int(y)), 20, color=(255, 0, 0))
    if 'x' not in locals():
        x = 0
    if 'y' not in locals():
        y = 0
    if 'w' not in locals() and 'h' not in locals():
        (h, w) = frame.shape[:2]
    
    return frame, x, y, h, w
