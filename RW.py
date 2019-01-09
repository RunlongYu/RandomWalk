# Implement Random Walk.
# Random Walk estimates the user's preference on an item via
# the average of all reachable users' preference on that item.
# @author Runlong Yu, Han Wu, Weibo Gao

from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
import scores
import Graph

class RW:
    user_count = 943
    item_count = 1682
    walk_length = 100
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    size_u_i = user_count * item_count
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)

    def load_data(self, path):
        user_ratings = defaultdict(set)
        max_u_id = -1
        max_i_id = -1
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
                max_u_id = max(u, max_u_id)
                max_i_id = max(i, max_i_id)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1

    def get_graph(self, user_ratings):
        temp_list = []
        for i in range(0, self.user_count - 1):
            for j in range(i + 1, self.user_count):
                count = 0
                for item in user_ratings[i]:
                    if item in user_ratings[j]:
                        count += 1
                if count > 10:
                    temp_list.append(str(i) + ' ' + str(j) + ' ' + str(count))
        with open('Graph.txt', 'w') as f:
            for line in temp_list:
                f.write(line + '\n')
        G = nx.read_edgelist('Graph.txt', nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        G = G.to_undirected()
        print ('len:', len(G.nodes()))
        return G

    def walk2predict(self, user_ratings, walks):
        predict_matrix = np.zeros((self.user_count, self.item_count))
        for users in walks:
            for item in range(1, self.item_count + 1):
                count = 0
                for i in range(1, len(users)):
                    if item in user_ratings[users[i]]:
                        count += 1
                predict_matrix[users[0] - 1][item - 1] = round(float(count) / self.walk_length, 10)
        return predict_matrix

    def train(self, user_ratings_train, walk_length):
        nx_G = self.get_graph(user_ratings_train)
        G = Graph.Graph(nx_G, 1, max(int(self.walk_length / 2), 1))
        G.process_transition_probs()
        walks = G.simulate_walks(1, walk_length + 1)
        predict_matrix = self.walk2predict(user_ratings_train, walks)
        return predict_matrix

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # training
        predict_matrix = self.train(user_ratings_train, self.walk_length)
        # prediction
        self.predict_ = predict_matrix.reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)

def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict

if __name__ == '__main__':
    rw = RW()
    rw.main()
