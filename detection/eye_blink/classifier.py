import keras,cv2,os,tensorflow as tf,numpy as np,logging,warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0
tf.config.experimental.set_memory_growth(physical_devices[0], True)
assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True#Could not create cudnn handle: CUDNN_STATUS_ALLOC_FAILED

import ipykernel#解决pycharm中keras进度条问题
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.densenet import DenseNet121,DenseNet169
from keras.preprocessing.image import ImageDataGenerator

#-------------------------------------------------------------
path='C:\\Users\\19051\\Desktop\\datasets\\(merged)postive_neg'
def make_val_set(path):
    read=os.path.join(path,'train')
    write=os.path.join(path,'validation')
    if not os.path.exists(write):
        os.mkdir(write)
    else:
        print('----------------------------------------------')
        print('Validation set already exists.Skipping creating now')
        return
    categories=os.listdir(read)
    for category in categories:
        save_dir=os.path.join(write,category)
        read_dir=os.path.join(read,category)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_names=os.listdir(read_dir)
        num_img=len(img_names)
        nums=np.random.choice(num_img,num_img//10,replace=False)
        for i in nums:
            img_path=os.path.join(read_dir,img_names[i])
            img=cv2.imread(img_path)
            cv2.imwrite(os.path.join(save_dir,img_names[i]),img)
            os.remove(img_path)
make_val_set(path)
batch_size=32
datagen=ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
)
val_gen=ImageDataGenerator()
train=datagen.flow_from_directory(
    os.path.join(path,'train'),
    target_size=(64,64),
    batch_size=32,
    interpolation='nearest',
    shuffle=True,
)
print(train.class_indices)
val=val_gen.flow_from_directory(
    os.path.join(path,'validation'),
    target_size=(64,64),
    batch_size=64,
    interpolation='nearest',
    shuffle=True,
)

# model=MobileNetV2(input_shape=(64,64,3),weights=None,classes=len(train.class_indices),pooling='avg',alpha=1)
model=DenseNet169(input_shape=(64,64,3),weights=None,classes=len(train.class_indices),pooling='avg',)
sgd=keras.optimizers.SGD(learning_rate=0.1,momentum=0.9,nesterov=True)
reduce_lr=keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',patience=5,min_delta=0.001,min_lr=1e-3,verbose=1)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train,steps_per_epoch=train.samples//batch_size,epochs=35,verbose=1,callbacks=[reduce_lr],validation_data=val)
model.save('Densenet_PN.hdf5')
