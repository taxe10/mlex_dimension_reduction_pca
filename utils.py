from pydantic import BaseModel, Field
import os
from PIL import Image
import numpy as np

class PCAParameters(BaseModel):
    n_components: int = Field(description='number of components to keep')


def load_images_from_directory(directory_path):
    image_data = []
    for filename in sorted(os.listdir(directory_path)):
        if filename.lower().endswith((".png", ".tif", ".tiff", ".jpeg", ".jpg")):
            file_path = os.path.join(directory_path, filename)
            try:
                img = Image.open(file_path)
                img_array = np.array(img)
                image_data.append(img_array)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    image_data = np.array(image_data)
    return image_data

# path = "/Users/runbojiang/Desktop/file_data_clinic/archive"
# load_images_from_directory(path)