import ultralytics
from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils import ops
from PIL import Image, ImageDraw, ImageFont
from ultralytics.utils.files import increment_path
import cv2
import numpy as np
import pafy
from copy import deepcopy
import concurrent.futures
from collections import defaultdict
import types
from pathlib import Path
import argparse
import torch
import torch.nn.functional as F
import time, os

from counter import Counter
from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

#------------------------------------------------------------------------------------------------------
# Arguments
#------------------------------------------------------------------------------------------------------
# get input argument
parser = argparse.ArgumentParser()
parser.add_argument('--webcam', action='store_true', help='use webcam')               # webcam usually is 0
parser.add_argument('--camera', type=int, default=None, help='camera port number')    # you can find it using find_port.py
parser.add_argument('--video-file', type=str, default=None, help='video filenames')   # example: "dataset_cam1.mp4"
parser.add_argument('--rtsp', type=str, default=None, help='rtsp link')               # example: "rtsp://192.168.1.136:8554/"
parser.add_argument('--youtube', type=str, default=None, help='youtube link')         # example: "http://www.youtube.com/watch?v=q0kPBRIPm6o"
parser.add_argument('--roi-xyxy', type=str, default=None, help='x1y1x2y2 of geofencing region of interest (in range 0 to 1), i.e.: [0.3,0.5,0.3,0.5]')
parser.add_argument('--stream-idx', type=int, default=0, help='Index for this video streaming')
opt = parser.parse_args()

# Define the source
WEBCAM = opt.webcam
CAMERA = opt.camera
VIDEO_FILE = opt.video_file
RTSP = opt.rtsp
YOUTUBE = opt.youtube # need ssl to be set

# Other arguments
ROI_XYXY   = opt.roi_xyxy
STREAM_IDX = opt.stream_idx
SAVE       = False
THRESH     = 0.35

#------------------------------------------------------------------------------------------------------
# Video streaming
#------------------------------------------------------------------------------------------------------
# load video source
def get_cap():
    if WEBCAM:
        cap = cv2.VideoCapture(0) # usually webcam is 0
    elif CAMERA is not None: 
        cap = cv2.VideoCapture(CAMERA)
    elif VIDEO_FILE:
        cap = cv2.VideoCapture(VIDEO_FILE)
    elif RTSP:
        cap = cv2.VideoCapture(RTSP)
    elif YOUTUBE:
        video = pafy.new(YOUTUBE)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)   
    else:
        assert False, "You do not specificy input video source!"
    return cap
    
cap = get_cap()

# resize your input video frame size (smaller -> faster, but less accurate)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
resize_width = 1280   # Adjust based on your needs
resize_height = 720  # Adjust based on your needs
if frame_width > 0:
    resize_height = int((resize_width / frame_width) * frame_height)



#------------------------------------------------------------------------------------------------------
# Overwrite original ultralytics function
#------------------------------------------------------------------------------------------------------
# to overwrite in ultralytics.engine.results.save_crop, save to stream index directory
def save_crop(self, save_dir, file_name=Path("im.png")):
    """
    Save cropped predictions to `save_dir/cls/file_name.png`.

    Args:
        save_dir (str | pathlib.Path): Save path.
        file_name (str | pathlib.Path): File name.
    """
    if self.probs is not None:
        LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
        return
    if self.obb is not None:
        LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
        return

    # variables
    boxes = self.boxes
    try:
        track_ids = self.boxes.id.int().cpu().tolist()
    except:
        track_ids = [None] * len(boxes)

    # save crops
    for d, id in zip(boxes, track_ids):
        if id is None:
            continue
        crop = save_one_box(
                            d.xyxy,
                            self.orig_img.copy(),
                            file=Path(save_dir) / f"stream_{STREAM_IDX}" / str(id) / f"{Path(file_name)}.png",
                            BGR=True,
                           )     
        # ReID
        crop = torch.from_numpy(crop)
        crop = crop.unsqueeze(0)
        crop = crop.permute(0, 3, 1, 2)


# to overwrite in ultralytics.engine.results.plot, plot Face ID instead of Track ID
def plot(
    self,
    reid_dict=None,
    conf=True,
    line_width=None,
    font_size=None,
    font="Arial.ttf",
    pil=False,
    img=None,
    im_gpu=None,
    kpt_radius=5,
    kpt_line=True,
    labels=True,
    boxes=True,
    masks=True,
    probs=True,
    show=False,
    save=False,
    filename=None,
):

    if img is None and isinstance(self.orig_img, torch.Tensor):
        img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

    names = self.names
    is_obb = self.obb is not None
    pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
    pred_masks, show_masks = self.masks, masks
    pred_probs, show_probs = self.probs, probs
    annotator = Annotator(
        deepcopy(self.orig_img if img is None else img),
        line_width,
        font_size,
        font,
        pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
        example=names,
    )

    # Plot Segment results
    if pred_masks and show_masks:
        if im_gpu is None:
            img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
            im_gpu = (
                torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                .permute(2, 0, 1)
                .flip(0)
                .contiguous()
                / 255
            )
        idx = pred_boxes.cls if pred_boxes else range(len(pred_masks))
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # Plot Detect results
    if pred_boxes is not None and show_boxes:
        for d in reversed(pred_boxes):
            c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
            if reid_dict is not None:
                id = reid_dict[id]
            name = ("" if id is None else f"id:{id} ") + names[c]
            label = (f"{name} {conf:.2f}" if conf else name) if labels else None
            box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
            annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

    # Plot Classify results
    if pred_probs is not None and show_probs:
        text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
        x = round(self.orig_shape[0] * 0.03)
        annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

    # Plot Pose results
    if self.keypoints is not None:
        for k in reversed(self.keypoints.data):
            annotator.kpts(k, self.orig_shape, radius=kpt_radius, kpt_line=kpt_line)

    # Show results
    if show:
        annotator.show(self.path)

    # Save results
    if save:
        annotator.save(filename)

    return annotator.result()
    

# to overwrite ultralytics.utils.plotting.save_one_box, save as png instead
def save_one_box(xyxy, im, file=Path("im.png"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    """
    Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop.

    This function takes a bounding box and an image, and then saves a cropped portion of the image according
    to the bounding box. Optionally, the crop can be squared, and the function allows for gain and padding
    adjustments to the bounding box.

    Args:
        xyxy (torch.Tensor or list): A tensor or list representing the bounding box in xyxy format.
        im (numpy.ndarray): The input image.
        file (Path, optional): The path where the cropped image will be saved. Defaults to 'im.png'.
        gain (float, optional): A multiplicative factor to increase the size of the bounding box. Defaults to 1.02.
        pad (int, optional): The number of pixels to add to the width and height of the bounding box. Defaults to 10.
        square (bool, optional): If True, the bounding box will be transformed into a square. Defaults to False.
        BGR (bool, optional): If True, the image will be saved in BGR format, otherwise in RGB. Defaults to False.
        save (bool, optional): If True, the cropped image will be saved to disk. Defaults to True.

    Returns:
        (numpy.ndarray): The cropped image.

    Example:
        ```python
        from ultralytics.utils.plotting import save_one_box

        xyxy = [50, 50, 150, 150]
        im = cv2.imread('image.png')
        cropped_im = save_one_box(xyxy, im, file='cropped.png', square=True)
        ```
    """

    if not isinstance(xyxy, torch.Tensor):  # may be list
        xyxy = torch.stack(xyxy)
    b = ops.xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = ops.xywh2xyxy(b).long()
    xyxy = ops.clip_boxes(xyxy, im.shape)
    crop = im[int(xyxy[0, 1]) : int(xyxy[0, 3]), int(xyxy[0, 0]) : int(xyxy[0, 2]), :: (1 if BGR else -1)]
    if save:
        file.parent.mkdir(parents=True, exist_ok=True)  # make directory
        f = str(increment_path(file).with_suffix(".png"))
        # cv2.imwrite(f, crop)  # save BGR, https://github.com/ultralytics/yolov5/issues/7007 chroma subsampling issue
        Image.fromarray(crop[..., ::-1]).save(f, quality=95, subsampling=0)  # save RGB
    return crop    

#------------------------------------------------------------------------------------------------------
# ReID function
#------------------------------------------------------------------------------------------------------
# reid
class ReIDManager:
    def __init__(self, chosen_model):
        self.feat_database = None
        self.total_unique_id = 0
        self.unique_track_ids = [-1] * 300
        self.track_to_match_id = {}
        self.track_to_match_alive = {}
        self.temp_id = {}
        self.alive_thresh = 5
        self.blur_ids = []
        self.prev_id = []        
        
        # temp variable
        temp = []
        
        #---------------------------------
        # embedding blur database x
        #---------------------------------
        # if got previous embedding database x
        if STREAM_IDX in [0, 4]:
            npy_file_path = f"{STREAM_IDX}_x.npy"
        else:
            npy_file_path = "0_x.npy" # we have not extracted the blur id for the remaining streams
        loaded_data = np.load(npy_file_path)         
        
        # define the blur ids
        if int(STREAM_IDX) == 0:                
            blur_ids = [6, 1, 5]
        elif int(STREAM_IDX) == 4:
            blur_ids = [1, 2, 9]
        else:
            blur_ids = [6, 1, 5] # dummy
        
        # get all blur ids as temp        
        for i in blur_ids:
            x = loaded_data[i]
            x = np.expand_dims(x, axis=0)
            temp.append(x)        
        
        #---------------------------------
        # embedding blur database y
        #---------------------------------        
        # if got previous embedding database y
        if STREAM_IDX in [0, 2, 4]:
            npy_file_path = f"{STREAM_IDX}_y.npy"
        else:
            npy_file_path = "0_y.npy" # we have not extracted the blur id for the remaining streams
        loaded_data = np.load(npy_file_path)    

        # define the blur ids
        if int(STREAM_IDX) == 0:                
            blur_ids = [5 ,6, 9, 11, 615]
        elif int(STREAM_IDX) == 2:
            blur_ids = [15, 192, 252]
        elif int(STREAM_IDX) == 4:
            blur_ids = [7, 8, 10, 11]
        else:
            blur_ids = [6, 1, 5] # dummy
               
        # get all blur ids as temp        
        for i in blur_ids:
            x = loaded_data[i]
            x = np.expand_dims(x, axis=0)
            temp.append(x)           

        #---------------------------------
        # combined blur databases
        #---------------------------------           
        # convert these ids to feat_database
        temp = np.concatenate(temp, axis=0)
        np.save(f"{STREAM_IDX}_combined.npy", temp)  
        self.feat_database = torch.tensor(temp)
        
        print(STREAM_IDX, self.feat_database.shape)
        
        # store the unblur ids for future reference
        self.blur_ids = [i for i in range(len(temp))]
        self.total_unique_id = len(self.blur_ids)
        

        #---------------------------------
        # combined database with registered image
        #---------------------------------        
        self.images, self.names = self.read_images_from_directory()
        self.feats = []
        for image in self.images:
            xyxy = np.array([[0, 0, image.shape[1], image.shape[0]]])
            feat = chosen_model.reid.get_features(xyxy, image)
            self.feats.append(feat)   
        self.feats = torch.from_numpy(np.concatenate(self.feats, axis=0))

        self.feat_database = torch.cat((self.feat_database, self.feats), axis=0)
        self.total_unique_id += len(self.images)
        
        self.feat_database = self.feats
        self.blur_ids = []
        

    def read_images_from_directory(self, directory_path='register', target_size=(112, 112), image_extensions=None):
        """
        Reads all images in the specified directory and returns them as a list of image arrays.
    
        Parameters:
        - directory_path (str): Path to the directory containing images.
        - image_extensions (list, optional): List of image file extensions to consider. 
          Defaults to common image formats if not provided.
    
        Returns:
        - images (list): List of images read from the directory as numpy arrays.
        """

        def extract_name_from_filename(s):
            parts = s.split('_')
            if len(parts) == 2:
                name, ext = parts[1].split('.')
            elif len(parts) == 1:
                name, ext = parts[0].split('.')
            return name        
        
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
        images = []
        
        # Iterate over all files in the directory
        names = []
        for filename in sorted(os.listdir(directory_path)):
            # Check if the file is an image by checking its extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # get name
                name = extract_name_from_filename(filename)
                names.append(name)
                # Construct full file path
                file_path = os.path.join(directory_path, filename)
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                # Append the image to the list
                if image is not None:
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # no need, reid_multibackend will do
                    image = cv2.resize(image, target_size)
                    images.append(image)
                else:
                    print(f"Warning: Couldn't read the image {file_path}")
    
        if len(images) >= 1:
            images = np.stack(images, axis=0)            
        else:
            assert len(self.images) >= 1, "Make sure there is registered image(s) in the register directory"
        
        return images, names
            
            
        
    def matching(self, track_ids, xyxys, img):
        # ReID model inference
        feats = chosen_model.reid.get_features(xyxys.numpy(), img)

        # normalize feats
        feats = torch.from_numpy(feats)
        feats = F.normalize(feats, dim=1)

        # init
        if self.feat_database is None:
            self.feat_database = feats
            for track_id in track_ids:
                self.total_unique_id += 1
                self.unique_track_ids.append(track_id)
                self.track_to_match_id[track_id] = self.total_unique_id
                self.track_to_match_alive[track_id] = 1
            return

        # cosine similarity scores, and matches
        '''
        Given that:
            self.feat_database is a M x 512 matrix
            self.feats is a N x 512 matrix
        
        M is the number of identities stored in database
        N is the number of detected faces in this frame
        
        Face matching is to perform a nested for loop of matching
        all M and N, which is M x N possibilities
        
        Instead of double for loop, we can do matrix multiplication:
            (M x 512) multiply with (512 x N)
        This will gives us M x N, which is all of the matching possibilities
        '''        
        cosine_sim = torch.mm(self.feat_database, feats.transpose(0, 1))
        #print(cosine_sim)
        match_ids        = torch.argmax(cosine_sim, dim=0).cpu().tolist()
        match_thresholds = torch.any(cosine_sim > THRESH, dim=0).cpu().tolist()
        
        #print(match_ids, match_thresholds)
        reid_dict = {}
        for idx, (track_id, match_id, match_threshold) in enumerate(zip(track_ids, match_ids, match_thresholds)):
            # if match
            if match_threshold:
                # check if is blur id
                if match_id in self.blur_ids:
                    reid_dict[track_id] = 'None'
                    self.track_to_match_id[track_id] = 'None'
                else:
                    #print(match_threshold)
                    reid_dict[track_id] = self.names[match_id]
                    self.track_to_match_id[track_id] = self.names[match_id]
            else:
                reid_dict[track_id] = 'None'
                self.track_to_match_id[track_id] = 'None'
        return reid_dict            
            


#------------------------------------------------------------------------------------------------------
# Functions for video streaming
#------------------------------------------------------------------------------------------------------
# get model
def get_model(opt):
    # overwrite function
    ultralytics.engine.results.Results.save_crop = save_crop 
    ultralytics.engine.results.Results.plot      = plot
    ultralytics.utils.plotting.save_one_box      = save_one_box
    
    # Load the YOLO model
    chosen_model = ultralytics.YOLO("yolov8m_face.pt")  # Adjust model version as needed

    # Load the ReID model
    chosen_model.reid = ReIDDetectMultiBackend(weights=Path("backbone_90000_vggface2.onnx"), device=torch.device(0), fp16=True)

    # ReID magager
    chosen_model.reid_manager = ReIDManager(chosen_model)

    # Load counter for geofencing based on ROI
    ROI_XYXY   = opt.roi_xyxy
    STREAM_IDX = opt.stream_idx
    if ROI_XYXY is not None:
        xyxy = ROI_XYXY.split(',')
        assert len(xyxy) == 4, 'xyxy should be 4 coordinates'
        xyxy = [float(item) for item in xyxy]
        x1, y1, x2, y2  = xyxy
        chosen_model.my_counter = Counter(x1, y1, x2, y2, STREAM_IDX)
    else:
        chosen_model.my_counter = None
        
    return chosen_model
    
    
# draw roi
def draw_roi(chosen_model, img):
    # img shape
    img_shape = img.shape
   
    # draw roi
    x1 = chosen_model.my_counter.roi_x1 * img_shape[1]
    y1 = chosen_model.my_counter.roi_y1 * img_shape[0]
    x2 = chosen_model.my_counter.roi_x2 * img_shape[1]
    y2 = chosen_model.my_counter.roi_y2 * img_shape[0]

    pts = [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
    pts = np.array(pts, int)
    pts = pts.reshape((-1, 1, 2))
    img = cv2.polylines(img, [pts], True, (0,0,255), 5)

    # put text
    text = f'in: {chosen_model.my_counter.count_in}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = int(img.shape[0] * 0.003)
    font_thickness = 2
    origin = (int(img.shape[0]*0.35), int(img.shape[1]*0.5))
    x, y = origin
    text_color = (255, 255, 255)
    text_color_bg = (0, 0, 0)
        
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    cv2.rectangle(img, origin, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img,text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img


# crop image
def crop_image(chosen_model, img):
    # img shape
    img_shape = img.shape
    
    # Convert normalized coordinates to absolute coordinates
    x1 = int(chosen_model.my_counter.big_roi_x1 * img_shape[1])
    y1 = int(chosen_model.my_counter.big_roi_y1 * img_shape[0])
    x2 = int(chosen_model.my_counter.big_roi_x2 * img_shape[1])
    y2 = int(chosen_model.my_counter.big_roi_y2 * img_shape[0])    
    
    # Crop the image
    cropped_img = img[y1:y2, x1:x2]
    
    return cropped_img

    
def paste_image(chosen_model, ori_img, cropped):
    # img shape
    img_shape = ori_img.shape
    
    # Convert normalized coordinates to absolute coordinates
    x1 = int(chosen_model.my_counter.big_roi_x1 * img_shape[1])
    y1 = int(chosen_model.my_counter.big_roi_y1 * img_shape[0])
    x2 = int(chosen_model.my_counter.big_roi_x2 * img_shape[1])
    y2 = int(chosen_model.my_counter.big_roi_y2 * img_shape[0])    
    
    cropped = cv2.resize(cropped, (int(x2-x1), int(y2-y1)))
    ori_img[y1:y2, x1:x2] = cropped
    return ori_img   


# predict
def predict(chosen_model, img, classes=[], conf=0.5):
    #resiz the image to 640x480
    img = cv2.resize(img, (resize_width, resize_height))
    if classes:
        results = chosen_model.track(img, classes=classes, conf=conf, save_txt=SAVE, persist=True, verbose=False, save_crop=SAVE)
    else:
        results = chosen_model.track(img, conf=conf, save_txt=SAVE, persist=True, verbose=False, save_crop=SAVE)

    return results


# predict and detect
def predict_and_detect(chosen_model, track_history, img, classes=[], conf=0.5):
    # resiz the image to 640x480
    img = cv2.resize(img, (resize_width, resize_height))
    img_shape = img.shape
    
    # make a copy
    img_ori = deepcopy(img)

    # crop
    img = crop_image(chosen_model, img)

    # get results           
    results = predict(chosen_model, img, classes, conf=conf)

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu()
    xyxys = results[0].boxes.xyxy.cpu()
    img = results[0].orig_img.copy()
    try:
        track_ids = results[0].boxes.id.int().cpu().tolist()
    except:
        # draw roi
        if chosen_model.my_counter is not None:  
            img = paste_image(chosen_model, img_ori, img)
            img = draw_roi(chosen_model, img)
            img = cv2.resize(img, (resize_width, resize_height))

        # log   
        return img, results

    # reid
    reid_dict = chosen_model.reid_manager.matching(track_ids, xyxys, img)

    # visualize
    annotated_frame = results[0].plot(reid_dict)
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 90 tracks for 90 frames
            track.pop(0)

        # Draw the tracking lines
        # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        # cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    if chosen_model.my_counter is not None:  
        # counter
        chosen_model.my_counter.update(img_shape, results[0])
    
        # draw roi
        annotated_frame = paste_image(chosen_model, img_ori, annotated_frame)
        annotated_frame = draw_roi(chosen_model, annotated_frame)
        img = cv2.resize(img, (resize_width, resize_height))
   
        # log
        chosen_model.my_counter.log()
   
    return annotated_frame, results


# process frame
def process_frame(track_history, frame):
    result_frame, _ = predict_and_detect(chosen_model, track_history, frame)
    return result_frame


# main
import time

def main():
    skip_frames = 2  # Number of frames to skip before processing the next one
    frame_count = 0  

    # Store the track history
    track_history = defaultdict(lambda: [])
    timestamp = [time.time()]
    framestamp = [0]
    
    # save video
    output_file = 'output_file.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 20
    out = cv2.VideoWriter(output_file, fourcc, output_fps, (resize_width, resize_height))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        cap = get_cap()
        while True:
            ret, frame = cap.read()
            time.sleep(0.020)
            if not ret:
                print(f'Stream {STREAM_IDX} Error fetching frame')                
                cap = get_cap()
                #break
                time.sleep(5)
                continue
            frame_count = 1+frame_count
            if frame_count % skip_frames != 0:
                continue  # Skip this frame

            # Submit the frame for processing
            future = executor.submit(process_frame, track_history, frame)
            result_frame = future.result()
            
            # record time and frame
            timestamp.append(time.time())
            framestamp.append(frame_count)
            
            # get delta
            if len(timestamp) < 30:
                idx1 = len(timestamp)-1
                idx2 = 0
            else:
                idx1 = len(timestamp)-1
                idx2 = len(timestamp)-30
                
            # get delta
            #print(timestamp[idx1], timestamp[idx2], idx1, idx2)
            delta_time = timestamp[idx1] - timestamp[idx2]
            delta_frame = framestamp[idx1] - framestamp[idx2]
            
            # get fps
            fps = delta_frame / delta_time

            # Display the processed frame
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow(f"Video Stream {STREAM_IDX}", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            #out.write(result_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    chosen_model = get_model(opt)
    main()
