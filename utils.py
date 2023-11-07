from pydantic import BaseModel, Field
import os
from PIL import Image
import numpy as np

class PCAParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')

def load_images_from_directory(directory_path):
    image_data = []
    for filename in os.listdir(directory_path)[:200]:
        if filename.endswith(".png"):
            file_path = os.path.join(directory_path, filename)
            try:
                # Open the image using Pillow
                img = Image.open(file_path)
                # Convert the image to a numpy array
                img_array = np.array(img)
                # Append the image array to the list
                image_data.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert the list of image arrays to a numpy array
    image_data = np.array(image_data)
    return image_data