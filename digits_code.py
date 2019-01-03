import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt 

(x_train,y_train),(x_test,y_test)=mnist.load_data()
test=x_test[0]
y_test[0]
plt.imshow(test)

x_train=x_train.reshape(x_train.shape[0],28,28,1)
x_test=x_test.reshape(x_test.shape[0],28,28,1)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

x_train /=255
x_test /=255

y_train=keras.utils.to_categorical(y_train,10)
y_test=keras.utils.to_categorical(y_test,10)

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'],loss=keras.losses.categorical_crossentropy)

model.fit(x_train,y_train,batch_size=128,epochs=12,validation_data=(x_test,y_test))

score=model.evaluate(x_test,y_test)
model.save('Final.model')

from keras.models import load_model
newModel=load_model('Final.model')
newModel.predict([x_test[0]])

