# Import all of the necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Open english text file and read each line
english_file = open("english.txt")
english_words = english_file.readlines()

# Open german text file and read each line
german_file = open("german.txt")
german_words = german_file.readlines()

# Open french text file and read each line
french_file = open("french.txt")
french_words = french_file.readlines()

# Define the arrays which will be needed to work with the data
char_to_ord_dataset = []
target_dataset = []
word_dataset = []

# For loop to iterate over the english words, filter to 5 letter words,
# append them to the right dataset, apply the ord function to convert the chars to ints,
# and then append a 0 to represent english words
for word in english_words:
    word = word.replace('\n', '')
    
    if len(word) == 5:
        word_dataset.append(word)
        char_to_ord_dataset.append([ord(char) for char in word])
        target_dataset.append(0)

# For loop to iterate over the german words, filter to 5 letter words,
# append them to the right dataset, apply the ord function to convert the chars to ints,
# and then append a 1 to represent german words
for word in german_words:
    word = word.replace('\n', '')
    
    if len(word) == 5:
        word_dataset.append(word)
        char_to_ord_dataset.append([ord(char) for char in word])
        target_dataset.append(1)

# For loop to iterate over the french words, filter to 5 letter words,
# append them to the right dataset, apply the ord function to convert the chars to ints,
# and then append a 2 to represent french words
for word in french_words:
    word = word.replace('\n', '')
    
    if len(word) == 5:
        word_dataset.append(word)
        char_to_ord_dataset.append([ord(char) for char in word])
        target_dataset.append(2)

# Combine all of the separate langauge data into one dataframe which has all the words together,
# along with their ord representations, and the target value for that word 
combined_df = pd.DataFrame({'Training(X)':word_dataset, 'Ord() representation': char_to_ord_dataset, 'Target(y)': target_dataset})

# use the train_test_split method to split the combined dataframe into training and testing data.
# The training data will comprise 80% of the data, while the other 20% will be for testing.
# The random_state value is included to ensure reproducibility across different "runs"
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)

# Define the different models that will be used for the assignment
knn_model = KNeighborsClassifier()
svm_model = svm.SVC()
mlp_nn = MLPClassifier()

# This is purely to provide a method to iterate over the methods in the loop below
models = [knn_model, svm_model, mlp_nn]

# For loop too iterate over the models, train the model with the training dataframe,
# make a prediction based on the testing dataframe, and then calculate the accuracy of the model
accuracies =[]
for model in models:
    model.fit(train_df['Ord() representation'].tolist(), train_df['Target(y)'].tolist())
    predictions = model.predict(test_df['Ord() representation'].tolist())
    print(type(model).__name__)
    print("Accuracy:", accuracy_score(test_df['Target(y)'].tolist(), predictions))
    accuracy = accuracy_score(test_df['Target(y)'].tolist(), predictions)
    accuracies.append(accuracy * 100)

# Plot the accuracies of the different models
labels = ("KNN", "SVM", "MLP")
value = accuracies
plt.title("Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Model")
plt.xlim(0, 100)
y_pos = np.arange(len(labels))
plt.barh(y_pos, value, align="center", alpha=0.5)
plt.yticks(y_pos, labels)
plt.show()
