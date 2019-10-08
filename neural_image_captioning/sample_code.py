from keras.layers import Input,merge
from keras.models import Model
import numpy as np

input_a=np.reshape([1,2,3],(1,1,3))
input_b=np.reshape([4,5,6],(1,1,3))

print(input_a)
print(input_b)

a=Input(shape=(1,3))
b=Input(shape=(1,3))


dot=merge([a,b],mode='dot',dot_axes=2)

mode_mot=Model(input=[a,b],output=dot)

print(mode_mot.predict([input_a,input_b]))