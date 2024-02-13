import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
MASK_THRESHOLD = 0.90 

def load_model(model_class, filepath, device):
    model = model_class
    model.load_state_dict(torch.load(filepath, map_location=device))
    return model

def get_model_instance_segmentation(num_classes):
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained = True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  # Replace the mask predictor with a new one
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,num_classes)
  return model

# Define the file path where your model is saved
saved_model_filepath = 'pytorch_model-e1.bin'

# Load the saved model
model = load_model(get_model_instance_segmentation(2), saved_model_filepath, 'cpu')
model.eval()
device = 'cpu'
# Now you can use 'loaded_model' to make predictions in real-time
input_video_path = "./instance.mkv"
output_video_path = "./instance_output.mkv"

# Open the webcam
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID') # Codec used for writing the video
out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height)) # Output file, codec, FPS, resolution

while (cap.isOpened()):
    # Read a frame from the webcam
    ret, frame = cap.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    HEIGHT, WIDTH, _ = rgb_image.shape
    
    # Perform inference
    image_tensor = torch.from_numpy(rgb_image.transpose((2, 0, 1))).float() / 255.0
    outputs = model([image_tensor])[0]
    print(outputs)

    # Create an empty mask image
    mask_image = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    # Combine masks
    for j, mask in enumerate(outputs['masks'].detach().numpy()):
        mask_image = np.logical_or(mask_image, mask[0] > MASK_THRESHOLD)

    # Convert mask image to uint8
    mask_image = mask_image.astype(np.uint8) * 255

    # Apply the mask onto the red mask
    preserved_content = cv2.bitwise_and(frame, frame, mask=mask_image)
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    blurred_content = cv2.bitwise_and(blurred_frame, blurred_frame, mask=cv2.bitwise_not(mask_image))

    # Apply the mask onto the original frame
    result = cv2.add(preserved_content, blurred_content)
    out.write(result)
    # Display the overlaid image
    cv2.imshow('Overlay', result)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
out.release()
cv2.destroyAllWindows()


