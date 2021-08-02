#K-Means clustering implementation

#Some hints on how to start, as well as guidance on things which may trip you up, have been added to this file.
#You will have to add more code that just the hints provided here for the full implementation.
#You will also have to import relevant libraries for graph plotting and maths functions.

import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn.metrics import pairwise_distances_argmin
from collections import Counter, defaultdict

# ====
# Define a function that reads data in from the csv files  
# HINT: http://docs.python.org/2/library/csv.html. 
# HINT2: Remember that CSV files are comma separated, so you should use a "," as a delimiter. 
# HINT3: Ensure you are reading the csv file in the correct mode.

def read_csv():
    
    x = [] #  BirthRate
    y = [] #  LifeExpectancy
    
    countries = []
    x_label = ""
    y_label = ""
    
    with open("dataBoth.csv") as csvfile:
        
        reader = csv.reader(csvfile, delimiter=',')
        lines = 0
        for row in reader:
            if lines >= 1:
                
               # print(', '.join(row))
                x.append(float(row[1]))
                y.append(float(row[2]))
                countries.append(row[0])
                lines += 1
            else:
                x_label = row[1]
                y_label = row[2]
               # print(', '.join(row))
                lines += 1
    return x, y, x_label, y_label, countries


# declare values to retrieve infomation in the csv file

x, y, x_label, y_label, countries = read_csv()


# ====
# 2a. Assign labels based on closest center
# I am using the pairwise_distances_argmin method to
# calculate distances between points to centres

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    # The main loop
    # This loop continues until convergence.


    
    while True:
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in
        range(n_clusters)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

  
    return centers, labels


#====
# declare number of clusters the *k*

clust_num = 4

#====
#combine x and y into a 2D list of (x, y) pairs

X = np.vstack((x, y)).T


#====
#visualise the result of our K-Means algorithm applied to the data
#chose Colormaps in Matplotlib
centers, labels = find_clusters(X, clust_num)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('K-Means clustering of countries by birth rate vs life expectancy')
plt.xlabel(x_label)
plt.ylabel(y_label)

# number of countries in each cluster
print("\nNumber of countries in each cluster:")
print(Counter(labels))
         
#====
# Get cluster indices
clusters_indices = defaultdict(list)
for index, c in enumerate(labels):
    clusters_indices[c].append(index)




# ====
# Print out the results for questions
#1.) The number of countries belonging to each cluster
#2.) The list of countries belonging to each cluster
#3.) The mean Life Expectancy and Birth Rate for each cluster
x = 0
while x < clust_num:
    print("\nCluster " + str(x + 1))
    print("************")
    for i in clusters_indices[x]:
        print(countries[i])
    print("************")
    print("Mean birth rate:")
    print(centers[x][0])
    print("Mean life expectancy:")
    print(centers[x][1])
    x+=1
    
plt.show()


