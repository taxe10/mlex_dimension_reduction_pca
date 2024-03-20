from sklearn.decomposition import PCA
import argparse
import pathlib
import numpy as np
import pandas as pd
import time
import yaml

from utils import load_images_from_directory

""" Compute PCA
    Input: 1d data (N, M) or 2d data (N, H, W)
    Output: latent vectors of shape (N, 2) or (N, 3)
"""

def computePCA(data, n_components=2, standarize=False):
    data = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str, help="path of yaml file for parameters")
    args = parser.parse_args()

    # Open the YAML file for all parameters
    with open(args.yaml_path, "r") as file:
        # Load parameters
        parameters = yaml.safe_load(file)

    # Validate and load I/O related parameters
    io_parameters = parameters["io_parameters"]
    # Check input and output dir are provided
    assert io_parameters["images_dir"], "Input dir (image filepath) not provided for training."
    assert io_parameters["output_dir"], "Output dir (dir to save the computed latent vactors) not provided for training."

    # Validate model parameters:
    model_parameters = parameters["model_parameters"]
    print("model_parameters")
    print(model_parameters)

    images_dir = io_parameters["images_dir"]
    output_dir = pathlib.Path(io_parameters["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Load images from given images_dir
    images = None
    if images_dir == 'data/example_shapes/Demoshapes.npz': # example dataset 
        images = np.load(images_dir)['arr_0']
    elif images_dir == 'data/example_latentrepresentation/f_vectors.parquet': # example dataset 
        df = pd.read_parquet(images_dir)
        images = df.values
    elif images_dir.split('.')[-1] == 'parquet': # data clinic
        df = pd.read_parquet(images_dir)
        images = df.values
    else: # user uploaded zip file
        images = load_images_from_directory(images_dir)
    print(images.shape)
    start_time = time.time()

    # Run PCA
    latent_vectors = computePCA(images, n_components=model_parameters['n_components'])
    print("Latent vector shape: ", latent_vectors.shape)
    
    # Save latent vectors
    output_name = 'latent_vectors.npy'
    np.save(str(output_dir) + '/' + output_name, latent_vectors)

    print("PCA done, latent vector saved.")
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution time: {execution_time} seconds")
