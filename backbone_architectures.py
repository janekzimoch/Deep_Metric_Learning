from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D,Input,Layer,Dense, Dropout, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Convolution2D, MaxPooling2D, concatenate, Lambda
from tensorflow.keras.layers import GlobalMaxPooling2D, ZeroPadding2D, Conv2D, BatchNormalization
from tensorflow.keras.layers import Add
from tensorflow.keras.initializers import glorot_uniform


from tensorflow.keras.regularizers import l2



def LeNet_5(input_shape=(64,64,3)):
    model_input = Input(shape=input_shape)
    
    activation = 'relu'
    x= Convolution2D(48,7,padding='same',activation=activation,name='CONV-1')(model_input)
    x= AveragePooling2D(pool_size=2,name='AVG-POOL-1')(x)
    x = Dropout(0.1, name='DROPOUT-1')(x)
    
    x= Convolution2D(96,5,padding='same',activation=activation,name='CONV-2')(model_input)
    x= AveragePooling2D(pool_size=2,name='AVG-POOL-2')(x)
    x = Dropout(0.1, name='DROPOUT-2')(x)

    x= Convolution2D(128, 5,padding='valid',activation=activation,name='CONV-3')(x)
    x= AveragePooling2D(pool_size=2,name='AVG-POOL-3')(x)
    x = Dropout(0.1, name='DROPOUT-3')(x)
    
    x= Convolution2D(160, 5,padding='valid',activation=activation,name='CONV-4')(x)
    x= AveragePooling2D(pool_size=2,name='AVG-POOL-4')(x)
    x = Dropout(0.1, name='DROPOUT-4')(x)

    x= Convolution2D(128, 5,padding='valid',activation=activation,name='CONV-5')(x)

    # prepare the final outputs
    pooledOutput = Flatten(name='FLATTEN')(x)
    outputs = Dense(64, name='IMAGE-EMBEDING')(pooledOutput)
    model = Model(model_input, outputs, name='LeNet_5') 
    return model



def VGG_16(embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5, L2_reg, drop_out, input_shape=(64,64,3)):
    model_input = Input(shape=input_shape)

    print('MODEL PARAMETERS: ')
    print(embedding_length, fc1, fc2, feat1,feat2,feat3,feat4,feat5)
    print(L2_reg, drop_out)
    print()

    # Block 1
    x = layers.Conv2D(feat1, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block1_conv1')(model_input)
    x = layers.Conv2D(feat1, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(feat2, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(feat2, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(feat3, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(feat3, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(feat3, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(feat4, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(feat4, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(feat4, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # , kernel_regularizer=l2(0.0005)
    # Block 5
    x = layers.Conv2D(feat5, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(feat5, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(feat5, (3, 3), kernel_regularizer=l2(L2_reg), activation='relu', padding='same', name='block5_conv3')(x)
    
    # prepare the final outputs
    x = Flatten(name='FLATTEN')(x)

    x = layers.Dense(fc1, activation='relu', name='fc1')(x)
    x = Dropout(drop_out, name='DROPOUT-1')(x)

    x = layers.Dense(fc2, activation='relu', name='fc2')(x)
    x = Dropout(drop_out, name='DROPOUT-2')(x)


    outputs = Dense(embedding_length, name='IMAGE-EMBEDING')(x)
    model = Model(model_input, outputs, name='LeNet_5') 
    return model




def VGG_16_mini(input_shape=(64,64,3)):
    model_input = Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block1_conv1')(model_input)
    x = layers.Conv2D(64, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(256, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), kernel_regularizer=l2(0.0005), activation='relu', padding='same', name='block5_conv2')(x)

    # prepare the final outputs
    x = Flatten(name='FLATTEN')(x)
    x = layers.Dense(2048, activation='relu', name='fc1')(x)
    x = layers.Dense(2048, activation='relu', name='fc2')(x)

    # x = Dropout(0.5, name='DROPOUT')(x)

    outputs = Dense(64, name='IMAGE-EMBEDING')(x)
    model = Model(model_input, outputs, name='LeNet_5') 
    return model




def ResNet_50(input_shape=(64,64,3)):
    model_input = Input(shape=input_shape)

   
    x = ZeroPadding2D((3, 3))(model_input)
    x = Conv2D(64, (7, 7), strides=(1, 1), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((8, 8), name='avg_pool')(x)
    
    # prepare the final outputs
    x = Activation('relu')(x)
    pooledOutput = Flatten(name='FLATTEN')(x)
    outputs = Dense(128, name='IMAGE-EMBEDING')(pooledOutput)
    
    model = Model(model_input, outputs, name='resnet50')
    
    return model



def ResNet_18(input_shape=(64,64,3)):
    model_input = Input(shape=input_shape)

   
    x = ZeroPadding2D((3, 3))(model_input)
    x = Conv2D(64, (7, 7), strides=(1, 1), name='conv1')(x)

    x = basic_res_block(x, 64)
    x = basic_res_block(x, 128)
    x = basic_res_block(x, 256)
    x = basic_res_block(x, 512)
    
    x = Activation('relu')(x)
    x = AveragePooling2D((4, 4), name='avg_pool')(x)
    
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    
    model = Model(model_input, x, name='resnet50')
    
    return model


def basic_res_block(inputs, filters):
    
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=0))(x)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(x)

    x = Conv2D(1, (1, 1), strides=(1, 1))(x)
    
    
    shortcut = Conv2D(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=glorot_uniform(seed=0))(inputs)

    out = Add()([shortcut, x])
    return out




def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


