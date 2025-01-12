# dds cloudapi for Grounding DINO 1.5
import copy
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.tasks.dinox import DinoxTask
from dds_cloudapi_sdk.tasks.types import DetectionTarget
from dds_cloudapi_sdk import TextPrompt

# media pipe for hand tracking
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import os
import cv2
import torch
import time
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from gpt_ability import get_instrument_category,get_movement
from get_frame_by_hands import process_frame_results,get_hand_motion,get_hand_speed

class human_motion_analysisor():
    def __init__(self):
        self.__Hyperparam_setting__()
        """
        Step 1: Environment settings and model initialization for SAM 2
        """
        # use bfloat16 for the entire notebook
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # init sam image predictor and video predictor model
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

    def __Hyperparam_setting__(self):
        """
        Hyperparam for Ground and Tracking
        """
        self.VIDEO_PATH = "../Result_Video2Ro_Chem/notebooks/videos/testvideo.mp4"
        self.TEXT_PROMPT = "test tube. beaker."
        self.OUTPUT_VIDEO_PATH = "../Result_Video2Ro_Chem/outputs/grounded_sam2_dinox_demo/testvideo_res.mp4"
        self.SOURCE_VIDEO_FRAME_DIR = "../Result_Video2Ro_Chem/custom_video_frames/testvideo"
        self.SAVE_TRACKING_RESULTS_DIR = "../Result_Video2Ro_Chem/tracking_results/testvideo"
        self.API_TOKEN_FOR_DINOX = "7956f36e628603116ef92c2aeb2d43c6"
        self.PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
        self.BOX_THRESHOLD = 0.2

        self.SLECTED_FOLDER = "../Result_Video2Ro_Chem/selected_frames/testvideo"
        self.BASE_FOLDER = Path("../Result_Video2Ro_Chem/speed_curve")
        if not os.path.exists(self.SLECTED_FOLDER):
            os.makedirs(self.SLECTED_FOLDER)
        if not os.path.exists(self.BASE_FOLDER):
            os.makedirs(self.BASE_FOLDER)

    def analyze_frame(self):
        '''
        Step 1: Environment settings and model initialization for SAM 2
        Step 2: Prompt DINO-X with Cloud API for box coordinates
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        Step 5: Visualize the segment results across the video and save them
        Step 6: Convert the annotated frames to video
        '''
        self.video_process()
        self.DINOX_object_detection()
        self.SAM_object_segement()
        self.visualize_result()
        self.get_key_frames()
        self.create_video()
        
    def __init_mediapipe__(self):
        '''
        Initialize MediaPipe Hands model
        '''
        # 初始化关于media pipe的手部追踪器参数
        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        num_frame = len(self.frame_names)
        self.all_landmark_pos = {"Right": np.zeros((num_frame, 2)), "Left": np.zeros((num_frame, 2))}

        # Create a hand landmarker instance with the video mode:
        self.mp_hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
            )
        
    def video_process(self):
        """
        Custom video input directly using video files
        """
        video_info = sv.VideoInfo.from_video_path(self.VIDEO_PATH)  # get video info
        print(video_info)
        frame_generator = sv.get_video_frames_generator(self.VIDEO_PATH, stride=1, start=0, end=None)

        # saving video to frames
        source_frames = Path(self.SOURCE_VIDEO_FRAME_DIR)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)

        # scan all the JPEG frame names in this directory
        self.frame_names = [
            p for p in os.listdir(self.SOURCE_VIDEO_FRAME_DIR)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # get the first frame to gerenate the chemistry instrument category
        img_path = os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[0])
        # cv2.imshow("frame", cv2.imread(img_path))
        self.TEXT_PROMPT = get_instrument_category(img_path)
        print("Chemistry instruments name:",self.TEXT_PROMPT)
        input("Press Enter to continue...")


        # init video predictor state
        self.inference_state = self.video_predictor.init_state(video_path=self.SOURCE_VIDEO_FRAME_DIR)

        self.ann_frame_idx = 0  # the frame index we interact with

    def DINOX_object_detection(self):
        """
        Step 2: Prompt DINO-X with Cloud API for box coordinates
        """

        # prompt grounding dino to get the box coordinates on specific frame
        img_path = os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[self.ann_frame_idx])
        image = Image.open(img_path)

        # Step 1: initialize the config
        config = Config(self.API_TOKEN_FOR_DINOX)

        # Step 2: initialize the client
        client = Client(config)

        # Step 3: run the task by DetectionTask class
        # image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
        # if you are processing local image file, upload them to DDS server to get the image url
        image_url = client.upload_file(img_path)

        task = DinoxTask(
            image_url=image_url,
            prompts=[TextPrompt(text=self.TEXT_PROMPT)],
            bbox_threshold=0.25,
            targets=[DetectionTarget.BBox],
        )

        client.run_task(task)
        result = task.result

        objects = result.objects  # the list of detected objects


        self.input_boxes = []
        self.confidences = []
        self.class_names = []

        for idx, obj in enumerate(objects):
            self.input_boxes.append(obj.bbox)
            self.confidences.append(obj.score)
            self.class_names.append(obj.category)

        self.input_boxes = np.array(self.input_boxes)

        print(self.input_boxes)

        # prompt SAM image predictor to get the mask for the object
        self.image_predictor.set_image(np.array(image.convert("RGB")))

        # process the detection results
        self.OBJECTS = self.class_names

        print(self.OBJECTS)

        # prompt SAM 2 image predictor to get the mask for the object
        self.masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=self.input_boxes,
            multimask_output=False,
        )
        # convert the mask shape to (n, H, W)
        if self.masks.ndim == 4:
            self.masks = self.masks.squeeze(1)

    def SAM_object_segement(self):
        """
        Step 3: Register each object's positive points to video predictor with seperate add_new_points call
        """

        assert self.PROMPT_TYPE_FOR_VIDEO in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"

        # If you are using point prompts, we uniformly sample positive points based on the mask
        if self.PROMPT_TYPE_FOR_VIDEO == "point":
            # sample the positive points from mask for each objects
            all_sample_points = sample_points_from_masks(masks=self.masks, num_points=10)

            for object_id, (label, points) in enumerate(zip(self.OBJECTS, all_sample_points), start=1):
                labels = np.ones((points.shape[0]), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels,
                )
        # Using box prompt
        elif self.PROMPT_TYPE_FOR_VIDEO == "box":
            for object_id, (label, box) in enumerate(zip(self.OBJECTS, self.input_boxes), start=1):
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    box=box,
                )
        # Using mask prompt is a more straightforward way
        elif self.PROMPT_TYPE_FOR_VIDEO == "mask":
            for object_id, (label, mask) in enumerate(zip(self.OBJECTS, self.masks), start=1):
                labels = np.ones((1), dtype=np.int32)
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=self.ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
        else:
            raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")

        """
        Step 4: Propagate the video predictor to get the segmentation results for each frame
        传播视频预测器以获得每帧的分割结果
        """
        self.video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def visualize_result(self):
        self.__init_mediapipe__()
        """
        Step 5: Visualize the segment results across the video and save them
        可视化视频中的分段结果并保存它们
        """

        if not os.path.exists(self.SAVE_TRACKING_RESULTS_DIR):
            os.makedirs(self.SAVE_TRACKING_RESULTS_DIR)

        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(self.OBJECTS, start=1)}


        frame_counter = 0


        for frame_idx, segments in self.video_segments.items():
            img = cv2.imread(os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[frame_idx]))

            '''
            Get the hand motion
            识别手部运动
            '''
            annotated_image, self.all_landmark_pos, Thumb, Index_finger = get_hand_motion(img,self.all_landmark_pos,self.HandLandmarker,self.mp_hand_options,frame_counter)
            frame_counter += 1
            
            all_object_ids = list(segments.keys())
            all_masks = list(segments.values())
            all_masks = np.concatenate(all_masks, axis=0)
            target_object_ids = []
            ID_TO_OBJECTS_copy = {i: obj for i, obj in enumerate(self.OBJECTS, start=1)}
            
            # get the coordinates of the bounding box
            boxes = sv.mask_to_xyxy(all_masks)
            # 判断手正在操作的对象
            target_ids = 1
            for box in boxes:
                if (Thumb[0] > box[0] and Thumb[0] < box[2]) and (Thumb[1] > box[1] and Thumb[1] < box[3]):
                    if (Index_finger[0] > box[0] and Index_finger[0] < box[2]) and (Index_finger[1] > box[1] and Index_finger[1] < box[3]):
                        # 添加文字“operating target”
                        ID_TO_OBJECTS_copy[target_ids] = ID_TO_OBJECTS[target_ids] + " Operating Target"
                        # print("Thumb and Index_finger in the box")
                target_ids += 1
                

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(all_masks),  # (n, 4)
                mask=all_masks, # (n, h, w)
                class_id=np.array(all_object_ids, dtype=np.int32),
            )

            # print("Boxes:",boxes)

            # visualize the segmentation results
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=annotated_image.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS_copy[i] for i in all_object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(self.SAVE_TRACKING_RESULTS_DIR, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)

        '''
        Replace the 0 in the landmark_pos with np.nan
        When one hand is detected, the other hand will have 0 as the x and y coordinates.
        '''
        for handedness in self.all_landmark_pos.keys():
            self.all_landmark_pos[handedness] = np.where(self.all_landmark_pos[handedness] == 0, np.nan, self.all_landmark_pos[handedness])

    def get_key_frames(self):
        '''
        Get the key frames of the video
        '''
        self.all_possible_handedness = set(self.all_landmark_pos.keys()) # all possible handedness detected in the video

        self.all_speeds = {}

        for handedness in self.all_possible_handedness:
            print(f"Calculating the speed curve of the {handedness} hand.")
            print(f"all the landmark pos: {self.all_landmark_pos[handedness]}")
            self.all_speeds[handedness] = get_hand_speed(self.all_landmark_pos[handedness])
            # print(f"Plotting the speed curve of the {handedness} hand.")
            # self.plot_speed(self.all_speeds[handedness], handedness)

        print("Deciding which hand to focus on.")
        self.handedness = "Right"
        print(f"The decided handedness is {self.handedness}.")

        print(f"Processing the speed curve of {self.handedness} hand.")
        smoothed_curve = self._process_speed_curve(handedness=self.handedness)

        print(f"Getting the peaks and valleys of the speed curve of {self.handedness} hand.")
        peaks, valleys = self._get_peaks_valleys(smoothed_curve)

        # Filter valleys based on the index difference
        selected_valleys = []
        for i in range(len(valleys)):
            if i == 0 or (valleys[i] - selected_valleys[-1]) >= 15:
                selected_valleys.append(valleys[i])
        
        # Save the selected valley frames
        img = cv2.imread(os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[0]))
        cv2.imwrite(f'{str(self.SLECTED_FOLDER)}/{0}.jpg', img)
        for valley in selected_valleys:
            img = cv2.imread(os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[valley]))
            cv2.imwrite(f'{str(self.SLECTED_FOLDER)}/{valley}.jpg', img)
        print(f"The selected valley frames are: {selected_valleys}")

        # for valley in valleys:
        #     img = cv2.imread(os.path.join(self.SOURCE_VIDEO_FRAME_DIR, self.frame_names[valley]))
        #     cv2.imwrite(f'{str(self.all_valleys_folder)}/{valley}.jpg', img)
        # print(f"All valley frames are: {valleys}")
        
    def _get_peaks_valleys(self, smoothed_curve):
        '''
        Get the peaks and valleys of the speed curve.

        Input: np.array. The smoothed speed curve.

        Return:
        peaks: np.array. The indices of the peaks.
        valleys: np.array. The indices of the valleys.
        '''
        peaks, _ = find_peaks(smoothed_curve)
        valleys, _ = find_peaks(-smoothed_curve)
        x = np.arange(len(smoothed_curve))

        plt.figure()
        plt.plot(x, smoothed_curve, label='Smoothed Data');
        plt.plot(x[peaks], smoothed_curve[peaks], 'rx', label='Peaks');
        plt.plot(x[valleys], smoothed_curve[valleys], 'go', label='Valleys');
        plt.title(f'Smoothed {self.handedness} Hand Speed Peaks and Valleys')
        plt.savefig(f'{str(self.BASE_FOLDER)}/{self.handedness}_speed_smoothed.jpg')
        plt.close()

        return peaks, valleys

    def _process_speed_curve(self, handedness):
        '''
        Process the speed curve of the hand so we can find peaks and valley more robustly.

        1. Linearly interpolate the nan values in the speed curve.
        2. Use Gaussian filter to smooth the speed curve.

        Return:
        np.array. The smoothed speed curve.
        '''

        # linear interpolation
        y = self.all_speeds[handedness]
        nans, x_nans = np.isnan(y), lambda z: z.nonzero()[0]
        y_interpolated = y.copy()
        y_interpolated[nans] = np.interp(x_nans(nans), x_nans(~nans), y[~nans])

        # Gaussian filter smoothing 
        y_smoothed = gaussian_filter(y_interpolated, sigma=5)

        return y_smoothed

    def _plot_speed(self, speeds, handedness, selected_frame=None):
        '''
        Plot the speed curve of the hand. Overlay the current hand speed of i-th frame on the entire speed curve.

        Input:
        speeds: np.array. The speed curve of the hand.
        handedness: str. The handedness of the hand. Either "Right" or "Left".
        '''
        for i in range(self.num_frame-1):
            '''
            For every frame, we are plotting the entire speed curve again and again for the purpose of making a video at the end.
            Again, can be better if I am more fluent in matplotlib.
            '''
            plt.figure()
            plt.plot(speeds, label=f'{handedness} Hand Speed Distribution')
            if selected_frame is not None:
                plt.scatter(selected_frame, speeds[selected_frame], color='blue', label=f'Selected Frame {selected_frame}')
            plt.scatter(i, speeds[i], color='red', label=f'Current {handedness} Hand')
            plt.legend()
            plt.savefig(f'{str(self.plot_folder)}/{i}_{handedness}.jpg')
            plt.close()


    def create_video(self):
        """
        Step 6: Convert the annotated frames to video
        """
        create_video_from_images(self.SAVE_TRACKING_RESULTS_DIR, self.OUTPUT_VIDEO_PATH)


def main():
    Human_Motion_Analysisor = human_motion_analysisor()
    Human_Motion_Analysisor.analyze_frame()

if __name__ == '__main__':
    main()