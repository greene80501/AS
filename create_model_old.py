c = input("WARNING!!! RUNNING THIS WILL CAUSE THE CURRENT MODEL TO BE ERASED AND RETRAINED!!! TYPE \"Y\" TO RUN THIS SCRIPT: ")
if c.lower() != 'y':
    exit()

import tensorflow as tf
import os, time
import numpy as np

x = []
y = []  # [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
read_s = time.time()

print("Reading letter data")

for letter_file in os.listdir("Corrected_CSV_Data"):
    with open("Corrected_CSV_Data/" + letter_file, 'r') as f:
        content = f.read()
        content = content.split('\n')
        
        for i, line in enumerate(content):
            if line != '' and line[0] != 'x':
                dats = line.split(',')
                print(len(dats))

                for i, dat in enumerate(dats):
                    if dat != '':
                        dats[i] = float(dat)
                
                x.append(dats)
                y.append(ord(letter_file[0]) - 65)

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

print(f"Letter data read in {time.time() - read_s} seconds")

num_classes = len(set(y))
y_encoded = tf.keras.utils.to_categorical(y, num_classes=num_classes)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(63,), activation='linear'),  # Ten dense layers with linear activation
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

"""for i in range(0, 10):
    model.add(tf.keras.layers.Dense(8))"""

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

c = input("LAST WARNING!!! RUNNING THIS WILL CAUSE THE CURRENT MODEL TO BE ERASED AND RETRAINED!!! TYPE \"Y\" TO RUN THIS SCRIPT: ")
if c.lower() != 'y':
    exit()

train_s = time.time()

try:
    model.fit(x, y_encoded, batch_size=32, epochs=100)
except Exception as e:
    print(f"Encountered exception: {e}")

print(f"Trained in {time.time() - train_s} seconds ({(time.time() - train_s) / 60} minutes)")

model.save("asl_model_idk.keras")
