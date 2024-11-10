# K-Nearest Neighbors (KNN) Implementation on the Iris Dataset

This Python script implements a simple K-Nearest Neighbors (KNN) classifier from scratch using the **Iris dataset**. The program predicts the class of test data points based on the nearest neighbors in the training set. Additionally, the script visualizes the dataset and provides an interactive scatter plot to inspect neighbors of selected points.

---

## Features

1. **KNN Classifier**:  
   A custom implementation of the KNN algorithm to classify Iris data points.

2. **Confusion Matrix**:  
   Visualizes the model's classification performance.

3. **Accuracy Metric**:  
   Calculates and displays the classification accuracy.

4. **Data Visualization**:  
   - Scatter plot of the Iris dataset.
   - Points are color-coded by class (Setosa, Versicolor, Virginica).

5. **Interactive Visualization**:  
   - Click on a point in the scatter plot to view its nearest neighbors and distances.

---

## Prerequisites

Ensure the following Python packages are installed:

- **NumPy**:  
  ```bash
  pip install numpy
  ```

- **Matplotlib**:  
  ```bash
  pip install matplotlib
  ```

- **Scikit-learn**:  
  ```bash
  pip install scikit-learn
  ```

---

## Usage

1. **Run the Script**:  
   Execute the script using Python:
   ```bash
   python knn_iris.py
   ```

2. **Understand the Output**:
   - **Confusion Matrix**:  
     A matrix is displayed showing the true vs. predicted classifications.

     ![image](https://github.com/user-attachments/assets/730a3547-6203-40fc-b553-058399c903e7)

   - **Accuracy**:  
     The percentage of correctly classified test points.

3. **Visualize the Data**:
   - A scatter plot of the Iris dataset is shown.
   - Each class is represented by a different color.
  
     ![image](https://github.com/user-attachments/assets/6d3b5065-4d01-4576-8ff4-da953d1fee34)


4. **Interact with the Scatter Plot**:
   - Click on a point to print its coordinates and the distances to its nearest neighbors in the terminal.

---

## Key Functions

- **`euclidean_distance(row1, row2)`**:  
  Computes the Euclidean distance between two points.

- **`get_neighbors(train, test_row, num_neighbors)`**:  
  Finds the `k` nearest neighbors of a test point.

- **`predict_classification(train, test_row, num_neighbors)`**:  
  Predicts the class of a test point based on its neighbors.

- **`plot_confusion_matrix(cm, classes, title)`**:  
  Visualizes the confusion matrix with color-coding.

---

## Dataset

The script uses the **Iris dataset**, a popular dataset containing measurements of three Iris species:
1. Setosa
2. Versicolor
3. Virginica

The dataset includes four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

---

## Interactive Features

### Clicking on Points
- Click on any point in the scatter plot.
- The terminal displays:
  - Coordinates of the clicked point.
  - Distances to its nearest neighbors.

---

## Visualization Example

- Scatter plot with Iris classes color-coded:
  - **Red**: Setosa
  - **Green**: Versicolor
  - **Blue**: Virginica

---

## Accuracy and Performance

The script calculates accuracy as:  
```text
Accuracy = (Number of Correct Predictions / Total Test Points) * 100
```

For the Iris dataset, accuracy is typically high due to the simplicity and separability of the data.

---

## Limitations

1. **Not Optimized for Large Datasets**:  
   - This implementation is designed for educational purposes and small datasets like Iris.
   - For larger datasets, consider using optimized libraries like Scikit-learn.

2. **Fixed Number of Neighbors**:  
   - The number of neighbors (`k`) is set to 3. You can modify this in the `num_neighbors` variable.

---

## Future Improvements

- Implement weighted KNN (e.g., weights inversely proportional to distance).
- Support additional distance metrics (e.g., Manhattan, Minkowski).
- Extend visualization to include all feature combinations.

---

Enjoy exploring the Iris dataset with KNN! 😊
