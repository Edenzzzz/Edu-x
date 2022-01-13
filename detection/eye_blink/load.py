from keras.models import  load_model
model=load_model('mobilenet_eye.hdf5')
model.summary()
print(model.input_shape)