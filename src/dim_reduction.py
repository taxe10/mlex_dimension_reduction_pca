from joblib import dump, load
from sklearn.decomposition import PCA


def compute_pca(
    data,
    n_components=2,
    load_model_path=None,
    save_model_path=None,
):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)

    if load_model_path:
        pca = load(load_model_path)
        data_pca = pca.transform(data)
    else:
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data)
        if save_model_path:
            dump(pca, save_model_path)
    return data_pca
