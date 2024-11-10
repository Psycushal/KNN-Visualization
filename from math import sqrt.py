from math import sqrt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import itertools

# This formula is for vectos
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row[:-1], train_row[:-1])
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append((distances[i][0], distances[i][1]))
    return neighbors

def predict_classification(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[0][-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

iris = load_iris()
data = iris.data
target = iris.target

dataset = [list(data[i]) + [target[i]] for i in range(len(data))]

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=1)

num_neighbors = 3
y_true = []
y_pred = []

for test_row in test_data:
    y_true.append(test_row[-1])
    prediction = predict_classification(train_data, test_row, num_neighbors)
    y_pred.append(prediction)

# Compute confusion matrix
num_classes = len(set(target))
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
for true, pred in zip(y_true, y_pred):
    confusion_matrix[true][pred] += 1

plt.figure()
plot_confusion_matrix(confusion_matrix, classes=iris.target_names, title='Confusion Matrix')
plt.show()

# Prints acc
correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
accuracy = correct / len(test_data) * 100
print(f'Accuracy: {accuracy:.2f}%')

# Visualise
fig, ax = plt.subplots(figsize=(8 / 2.54, 8 / 2.54))  # Is not working idk
colors = ['red', 'green', 'blue']
labels = iris.target_names

sc = []
for i, label in enumerate(labels):
    sc.append(ax.scatter(data[target == i, 0], data[target == i, 1], color=colors[i], label=label))

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset Visualization')
plt.legend()

#############################################################
# Event handler for clicking on points
def on_pick(event):
    ind = event.ind[0]
    clicked_point = dataset[ind]
    neighbors = get_neighbors(dataset, clicked_point, 3)
    distances = [neighbor[1] for neighbor in neighbors]
    print(f'Clicked Point: {clicked_point[:-1]}')
    for i, dist in enumerate(distances):
        print(f'Distance to neighbor {i+1}: {dist:.2f}')

# Connect the event handler
fig.canvas.mpl_connect('pick_event', on_pick)

# Make points selectable
for s in sc:
    s.set_picker(True)
#############################################################

plt.show()