from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np

class Clusterer:
    def __init__(self, method='kmeans', **kwargs):
        self.method = method
        self.model = None
        self.kwargs = kwargs
        
    def fit_predict(self, data):
        if self.method == 'kmeans':
            self.model = KMeans(**self.kwargs)
        elif self.method == 'dbscan':
            self.model = DBSCAN(**self.kwargs)
        elif self.method == 'agglomerative':
            self.model = AgglomerativeClustering(**self.kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
            
        return self.model.fit_predict(data)
