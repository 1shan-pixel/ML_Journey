import tensorflow as tf 
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import os 

digits = tf.keras.datasets.mnist

(X_train , y_train) , (X_test, y_test) = digits.load_data()





X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis =1 )
'''

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape= (28,28)))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(128, activation = 'relu'))
model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train , epochs = 5)

model.save('digits_model.keras')



'''

new_model = tf.keras.models.load_model('digits_model.keras')

# print(model.evaluate(X_test, y_test)) print(model.evaluate(X_test, y_test))
# tives out loss and accuracy so best way to do is 

#loss , accuracy = model.evaluate(X_test, y_test)


image_no = 1 

while os.path.isfile(f"digits/digit{image_no}.jpg"):
    try: 
        img = cv2.imread(f"digits/digit{image_no}.jpg")[:,:,0]
        img = cv2.resize(img,(28,28))
        img = np.invert(np.array([img]))
        predicition = np.argmax(new_model.predict(img))
        print("The digit is probably a  : ", predicition)
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except Exception as e: 
        print(e)
    finally : 
        image_no += 1

