from collections import Counter

             

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y, split_pct=None): 
        """
        Args :
        ----
        split_pct : (float) default None take values from 0 to 1
         if None all data used for evaluating the distance ,otherwise
         the data splited by precentage
          
        """
        if split_pct is None:
            self.x_train = X
            self.y_train = y
        else:    
            self.x_train, self.y_train, self.x_test, self.y_test = \
                        self.train_test_split(X, y, test_size=split_pct)
            return self.x_test, self.y_test


    def predict(self, X, distance_method=None):
        """this function compute the distance between given value "x" and each data point in the dataset

            choose shortest k values the most_common label from those k values  is choosen as label for the value "x"
            
            Args:
            ----
            X : list of test data points
            distance_method : 'euclidean_distance' or 'manhtin_distance'

            """
        predicted_labels = [self._predict(x, distance_method) for x in X]
        return predicted_labels


    def _predict(self, x, dist_method=None):

        # compute the distance
        dist_method = self.euclidean_distance if dist_method is None else dist_method

        # calculate distance between the "point x" and each point in teh dataset
        distances = [(i, dist_method(x, data_point)) for i, data_point in enumerate(self.x_train)]
        # sort acorroding to the distance
        distances.sort(key=lambda x:x[1])
        
        # get the indices of the nearest nighbors
        k_neighbors_indices = [i for i, _ in distances]
        k_neighbors_indices = k_neighbors_indices[:self.k]

        k_neighbors_labels = [self.y_train[idx] for idx in k_neighbors_indices]
        most_common_neighbor = self.most_common(k_neighbors_labels)
        return most_common_neighbor[0][0]


    def train_test_split(self, X, y, test_size):
        pct = len(X) - (len(X)* test_size)
        pct = int(pct)
        trainX, testX = X[:pct] ,X[pct:] 
        trainy, testy = y[:pct] ,y[pct:]
        return (trainX, trainy, testX, testy)



    def euclidean_distance(self, x1, x2):
        sum_res = 0
        for p1, p2 in zip(x1, x2):
            sum_res += (p1 - p2)**2
        res = sum_res**(1/2) # square root
        return res


    def manhatin_distance(self, x1, x2):
        res = 0
        for p1, p2 in zip(x1, x2):
            res += self.abs(p1 - p2)
        return res
    
    @staticmethod
    def most_common(input_list):
        itms_count = dict()
        for itm in input_list:
            if itm not in itms_count.keys():
                itms_count[itm] = 1
            itms_count[itm] += 1
        most_common_itm = sorted(itms_count.items(), key=lambda x:x[1],reverse=True)
        return most_common_itm

    
    @staticmethod
    def abs(value):
        if value < 0:
            value *= -1
        return value 


