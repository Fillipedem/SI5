import numpy as np

class Kmeans():

    def __init__(self, data, k=2, centroids=None):
        # data
        self.__check_numpy(data)
        self.data = data

        # centroids/k
        if centroids:
            self.__check_numpy(centroids)
            self.centroids = centroids
            self.k = len(self.centroids)
        else: # genereta random centroids
            self.centroids = self.__generate_random_centroids(k)
            self.k = k

        self.history = [self.centroids]

    def __check_numpy(self, np_array):
        """
        Checks if np_array is as numpy array with dimensions equal to 2
        input: np_array
        """

        if type(np_array) is not np.ndarray:
            raise ValueError("Data argument is not a numpy array!!")

        if np_array.ndim != 2:
            raise ValueError("Numpy array has dimensions diff from 2!!")


    def __generate_random_centroids(self, k):
        """
        generate new random centroids
        """
        min_value, max_value = np.amin(self.data, axis=0), np.amax(self.data, axis=0)
        vector_lenght = len(self.data[0])

        centroids = min_value + np.random.rand(k, vector_lenght)*(max_value - min_value)
        return centroids


    def __distace(self, a, b):
        """
        a, b: numpy array
        return: Euclidean distance
        """
        return np.sqrt(np.dot(a - b, a - b))

    def one_step(self):
        """
        Do one kmeans step
        """
        # calculando para cada  ponto da base de dados
        # qual cluster ele pertence
        clusters = {}
        for i in range(len(self.centroids)):
            clusters[i] = []

        for i in range(len(self.data)):
            min_distance = float('inf')
            closest_centroid = 0

            for j in range(len(self.centroids)):
                distance = self.__distace(self.data[i], self.centroids[j])

                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = j
            #endfor
            clusters[closest_centroid].append(i)
        #endfor

        # calculando novos centroids
        new_centroids = np.zeros(self.centroids.shape)

        for c in clusters:
            count = 0
            for i in clusters[c]:
                new_centroids[c] += self.data[i]
                count +=1

            if count:
                new_centroids[c] /= count
            else:
                new_centroids[c] = self.centroids[c]

        self.history.append(new_centroids)
        self.centroids = new_centroids
        
        return clusters
   
