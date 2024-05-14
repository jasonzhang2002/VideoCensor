from fastsam import FastSAM, FastSAMPrompt
import cv2
import numpy as np
from PIL import Image, ImageFilter
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip

def video_to_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")

    # Initialize list to store frames
    frames = []

    # Read the video frame by frame
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()

        # Break the loop if there are no more frames
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)

    print(f"Extracted {len(frames)} frames")

    return frames


def frames_to_video(pixelated_images, video):
    video_clip = ImageSequenceClip(pixelated_images, fps=24)  # Set the FPS as needed

    # Load the audio file
    audio_clip = AudioFileClip("/content/FastSAM/demo_shortened.mp3")  # Replace with the path to your audio file

    # Combine the video clip and audio clip
    final_clip = video_clip.set_audio(audio_clip)

    # Path to save the output video file
    output_video_path = video

    # Write the final combined video to a file
    final_clip.write_videofile(output_video_path, codec='libx264')  # Specify codec as needed

    # Close the video clips
    video_clip.close()
    audio_clip.close()
    final_clip.close()

    print("Video with audio created successfully:", output_video_path)



def pixelate_image(annotations, image, pixel_size, show=False):
    annotations = np.array(annotations)

    pixelated_image = np.copy(image)  # Create a copy of the original image to preserve it
    is_pixelated = np.zeros_like(annotations, dtype=bool)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Check if the corresponding pixel in the boolean array is True
            if annotations[y, x] and not is_pixelated[y, x]:
                # Pixelate the pixel
                diff = pixel_size // 2
                pixelated_image[y-diff:y+diff, x:x+diff] = cv2.resize(image[y-diff:y+diff, x:x+diff], (1, 1), interpolation=cv2.INTER_NEAREST)
                is_pixelated[y-diff:y+diff, x-diff:x+diff] = True

    if show:
        plt.figure(figsize=(10, 10))
        if (pixelated_image == True).any():
            plt.imshow(pixelated_image)
            plt.axis('off')  # Turn off axis
            plt.show()
          
    return pixelated_image


class ImageCensor():
    def __init__(self, weights, device, prompts, retina=True, imgsz=1024, conf=0.4, iou=0.8):
        self.model = FastSAM(weights)
        self.model.to(device)
        self.device = device
        self.prompts = prompts
        self.retina = retina
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.prev_image = None
        self.prev_mask = None
        self.frames_skipped = 0
    
    def compare_images(self, image, prev_image):

        # Compute absolute difference between the two images
        diff = cv2.absdiff(image, prev_image)

        # Convert difference image to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        gray_diff = np.sum(gray_diff)

        print("diff: " + str(gray_diff / (image.shape[0] * image.shape[1])))

        if gray_diff < image.shape[0] * image.shape[1] * 25:
            return True

        return False

    def test_pixelate(self, image):
        input = cv2.imread(image)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        everything_results = self.model(
                input,
                device=self.device,
                retina_masks=self.retina,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou
                )

        prompt_process = FastSAMPrompt(input, everything_results, device=self.device)
      
        ann = [False * input.shape[0]] * input.shape[1]

        for prompt in self.prompts:
            annotations = prompt_process.text_prompt(text=prompt)
            for annotation in annotations:
                ann = ann | annotation

        pixelate_image(ann, input, 30, show=True)

    def process_video(self, video_path):
        frames = video_to_frames(video_path + ".webm")
        print(end - start)

        processed_frames = []

        count = 0

        for frame in frames:
            print("count: " + str(count))

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = None

            if count >= 1 and self.frames_skipped < 30 and self.compare_images(frame, self.prev_image):
                mask = self.prev_mask
                self.frames_skipped += 1
            
            else:
                self.frames_skipped = 0
                everything_results = self.model(
                    frame,
                    device=self.device,
                    retina_masks=self.retina,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou
                    )

                prompt_process = FastSAMPrompt(frame, everything_results, device=self.device)

                mask = [False * frame.shape[0]] * frame.shape[1]

                for prompt in self.prompts:
                    annotations = prompt_process.text_prompt(text=prompt)
                    for annotation in annotations:
                        mask = mask | annotation

                self.prev_mask = mask
                self.prev_image = frame

            processed_frames.append(pixelate_image(mask, frame, 30, show=False))

            count += 1

        start = time.time()
        frames_to_video(processed_frames, video_path + '_blurred.mp4')
        end = time.time()

        return processed_frames
