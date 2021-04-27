from keras.datasets import cifar10
from keras import utils
from keras.preprocessing.image import ImageDataGenerator

def load_cifar10():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_y = utils.to_categorical(train_y)
    test_y = utils.to_categorical(test_y)
    return (train_x, train_y), (test_x, test_y)

def train_val_generators():
    train_datagen = ImageDataGenerator(rescale=1/255, horizontal_flip=True, brightness_range=[0.8, 1.1], rotation_range=5, zoom_range=[0.95, 1.1])
    val_datagen = ImageDataGenerator(rescale=1/255.)
    return train_datagen, val_datagen

def train_val_iterators(train_x, train_y, test_x, test_y, train_gen, val_gen):
    train_it = train_gen.flow(train_x, train_y, batch_size=32, shuffle=True)
    val_it = val_gen.flow(test_x, test_y, batch_size=100)
    train_steps = train_it.n//train_it.batch_size
    val_steps = val_it.n // val_it.batch_size
    return train_it, train_steps, val_it, val_steps
    # train_generator = datagen.flow_from_dataframe(dataframe=train_df, directory=train_dir,
    #                                          x_col='Id',
    #                                          y_col='Category',
    #                                          target_size=(150, 150),
    #                                          class_mode='categorical',
    #                                          batch_size=32,
    #                                          seed=7)
    # validation_generator = val_datagen.flow_from_dataframe(dataframe=val_df, directory=train_dir,
    #                                             x_col='Id',
    #                                             y_col='Category',
    #                                             shuffle=False,
    #                                             target_size=(150, 150),
    #                                             class_mode='categorical',
    #                                             batch_size=100,
    #                                             seed=7)
