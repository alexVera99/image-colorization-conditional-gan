import ganColorization.utils as UL
from PIL import Image
import torchvision.transforms as tf
import torch

def modelPredictWeb(file_path, model):
    # Define transformations
    to_tensor = tf.ToTensor()
    to_normalize = tf.Normalize(mean = [.5], std = [.5])
    to_resize = tf.Resize([96,96])
    to_pil = tf.ToPILImage(mode="RGB")
    
    # Open image
    rgba_img = Image.open(file_path)
    rgb_img = rgba_img.convert(mode="RGB")
    
    # To tensor 
    rgb_img = to_tensor(rgb_img)
    
    # Resize
    #rgb_img = to_resize(rgb_img)
    
    # Obtain gray image
    gray_img = tf.functional.rgb_to_grayscale(rgb_img)
    
    # Create mask
    mask = UL.createMask(rgb_img)
    
    # Normalizing
    gray_img = to_normalize(gray_img)
    mask = to_normalize(mask)
    
    # Concatenate gray image and mask
    img_mask = torch.concat([gray_img, mask], dim=0)
    
    # Add the batch dimension
    img_mask = img_mask.unsqueeze(0)
    
    # Compute the prediction (colorizing)
    color_img = model.forward(img_mask)
    
    # Converting from 0 to 1
    color_img = UL.imageIn0to1Torch(color_img)
    
    # COnvert to pil image
    result = to_pil(color_img.squeeze())
    
    # Save the image to show it later on
    result.save("./static/img_results/result.jpg")
    
    # Save the original image with 96x96 resolution
    original = to_pil(rgb_img)
    original.save("./static/img_results/original_96_96.jpg")
    
    return {"response": True}