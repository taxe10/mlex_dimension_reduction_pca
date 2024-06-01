from sklearn.decomposition import PCA


def compute_pca(data, n_components=2):
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca
