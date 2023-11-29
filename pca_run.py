from sklearn.decomposition import PCA
import argparse
import pathlib
import numpy as np
import json
import pandas as pd
import time

from utils import PCAParameters, load_images_from_directory

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

    parser.add_argument('image_dir', help='image filepath')
    parser.add_argument('output_dir', help='dir to save the computed latent vactors')
    parser.add_argument('parameters', help='dictionary that contains model parameters')
    
    args = parser.parse_args()

    images_dir = args.image_dir
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ## Load images from given images_dir
    images = None
    if images_dir == 'data/example_shapes/Demoshapes.npz':
        images = np.load(images_dir)['arr_0']
    elif images_dir == 'data/example_latentrepresentation/f_vectors.parquet':
        df = pd.read_parquet(images_dir)
        images = df.values
    else: # user uploaded zip file
        images = load_images_from_directory(images_dir)
    print(images.shape)
    start_time = time.time()
    # Load dimension reduction parameter
    if args.parameters is not None:
        parameters = PCAParameters(**json.loads(args.parameters))
    
    print(f'PCA parameters: {parameters.n_components}')

    # Run PCA
    latent_vectors = computePCA(images, n_components=parameters.n_components)
    print("Latent vector shape: ", latent_vectors.shape)
    
    # Save latent vectors
    output_name = 'latent_vectors.npy'
    np.save(str(output_dir) + '/' + output_name, latent_vectors)

    print("PCA done, latent vector saved.")
    end_time = time.time()
    execution_time = end_time - start_time

    # Print the execution time
    print(f"Execution time: {execution_time} seconds")
