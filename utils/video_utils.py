import cv2
import os
from tqdm import tqdm

def create_video_from_images(image_folder,plot_folder, output_video_path, frame_rate=25):
    # define valid extension
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()  # sort the files in alphabetical order
    # print(image_files)
    plot_image_files = [f for f in os.listdir(plot_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    plot_image_files.sort()
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    if not plot_image_files:
        raise ValueError("No valid image files found in the specified folder.")
    
    # load the first image to get the dimensions of the video
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape
    
    # create a video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # codec for saving the video
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width+500, height))
    
    # write each image to the video
    for i in tqdm(range(len(image_files)-1)):
        image_path = os.path.join(image_folder, image_files[i])
        image = cv2.imread(image_path)
        plot = cv2.imread(f'{plot_folder}/{i}_Smoothed Right.jpg')
        plot = cv2.resize(plot, (500, height))
        combined = cv2.hconcat([image, plot])
        video_writer.write(combined)
    
    # source release
    video_writer.release()
    print(f"Video saved at {output_video_path}")

