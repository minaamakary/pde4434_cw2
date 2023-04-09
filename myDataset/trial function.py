import os
import cv2
from cv2 import cuda_BufferPool # import cv2
#from cv2 import cuda_BufferPool 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_image_data():
    X = []
    y = []
    for folder_name in os.listdir('/Users/minamakary/Documents/pde4434_cw2/myDataset/b0'):
        folder_path_full = os.path.join('/Users/minamakary/Documents/pde4434_cw2/myDataset/b0', folder_name)
        if not os.path.isdir(folder_path_full):
            continue
        for image_name in os.listdir(folder_path_full):
            image_path = os.path.join(folder_path_full, image_name)
            try:
                # Load image and resize to 224x224 pixels
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (224, 224))
                # Append image data and label to X and y
                X.append(img)
                y.append(folder_name)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    X = np.array(X)
    y = np.array(y)
    print(X)

    return X, y

#code runs correctly until here


#MACHINE LEARNING 
X,y = load_image_data()
print(X)


# Preprocess data
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Choose a model and train it
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

