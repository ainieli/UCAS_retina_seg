from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, \
    Conv2DTranspose, Conv2DTranspose, concatenate
from keras.optimizers import Adam
from keras.regularizers import l2

from Dice import DiceLoss


########## 模型定义部分 ##########
# 定义U-Net模型
# 由于U-Net接受的输入是三维的，实际图像是二维的，故input_size需要扩维,
# 而input_size是tuple，故用 xxx + (1,)表示在末尾扩一维，其中xxx是一个二元组
def U_Net(input_size=(256, 256, 3), lr=1e-4, wd=1e-4):
    # 特征提取（4次下采样）
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv4)
    drop4 = Dropout(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv5)
    drop5 = Dropout(0.2)(conv5)

    # 图像还原（4次上采样）
    #up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(drop5)
    merge6 = concatenate([drop4, up6], axis=3)  # 跳层连接，将上采样得到的结果与对称步骤的卷积结果进行拼接
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv6)

    #up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv7)

    #up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv8)

    #up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv9)

    # 激活
    conv10 = Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv9)

    # 编译模型
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy', DiceLoss])

    # 打印模型总结
    model.summary()
    return model


# 定义U-Net++模型
def U_Net_plus_plus(input_size=(256, 256, 3), deep_supervision=False, lr=1e-4, wd=1e-4):
    inputs = Input(input_size)
    conv0_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(inputs)
    conv0_0 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv0_0)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

    conv1_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool1)
    conv1_0 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_0)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

    up1_0 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_0)
    merge00_10 = concatenate([conv0_0, up1_0], axis=-1)
    conv0_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge00_10)
    conv0_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv0_1)

    conv2_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool2)
    conv2_0 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_0)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

    up2_0 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_0)
    merge10_20 = concatenate([conv1_0, up2_0], axis=-1)
    conv1_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge10_20)
    conv1_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_1)

    up1_1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_1)
    merge01_11 = concatenate([conv0_0, conv0_1, up1_1], axis=-1)
    conv0_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge01_11)
    conv0_2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv0_2)

    conv3_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool3)
    conv3_0 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv3_0)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

    up3_0 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv3_0)
    merge20_30 = concatenate([conv2_0, up3_0], axis=-1)
    conv2_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge20_30)
    conv2_1 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_1)

    up2_1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_1)
    merge11_21 = concatenate([conv1_0, conv1_1 ,up2_1], axis=-1)
    conv1_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge11_21)
    conv1_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_2)

    up1_2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_2)
    merge02_12 = concatenate([conv0_0, conv0_1, conv0_2, up1_2], axis=-1)
    conv0_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge02_12)
    conv0_3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv0_3)

    conv4_0 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(pool4)
    conv4_0 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv4_0)

    up4_0 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv4_0)
    merge30_40 = concatenate([conv3_0, up4_0], axis = -1)
    conv3_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge30_40)
    conv3_1 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv3_1)

    up3_1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv3_1)
    merge21_31 = concatenate([conv2_0, conv2_1, up3_1], axis=-1)
    conv2_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge21_31)
    conv2_2 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_2)

    up2_2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv2_2)
    merge12_22 = concatenate([conv1_0, conv1_1, conv1_2, up2_2], axis=-1)
    conv1_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge12_22)
    conv1_3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_3)

    up1_3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv1_3)
    merge03_13 = concatenate([conv0_0, conv0_1, conv0_2, conv0_3, up1_3], axis=-1)
    conv0_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(merge03_13)
    conv0_4 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(wd))(conv0_4)

    # 激活
    nestnet_output_1 = Conv2D(1, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal', kernel_regularizer=l2(wd),
                              padding='same')(conv0_1)
    nestnet_output_2 = Conv2D(1, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal', kernel_regularizer=l2(wd),
                              padding='same')(conv0_2)
    nestnet_output_3 = Conv2D(1, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal', kernel_regularizer=l2(wd),
                              padding='same')(conv0_3)
    nestnet_output_4 = Conv2D(1, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal', kernel_regularizer=l2(wd),
                              padding='same')(conv0_4)

    # 深监督决定输出
    # model = Model(input=inputs, output=conv0_4)
    if deep_supervision:
        model = Model(input=inputs, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=inputs, output=[nestnet_output_4])

    # 编译模型
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy', DiceLoss])

    # 打印模型总结
    model.summary()
    return model