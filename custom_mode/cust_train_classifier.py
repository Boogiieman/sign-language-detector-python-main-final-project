import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the data dictionary
data_dict = pickle.load(open('custom_mode\cust_data.pickle', 'rb'))

# Pad sequences
data = pad_sequences(data_dict['data'], dtype='float32', padding='post')

# Convert labels to NumPy array
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

# Print the accuracy score
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model to a file
with open('regional\Malayalam\mal_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

"""
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences #added

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = pad_sequences(data_dict['data'], dtype='float32', padding='post') #added

#data = np.asarray(data_dict['data'])#commented to check
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
"""