"""
    完成U-Net在DRIVE视网膜血管图像上的训练
"""
import cv2
import os
import numpy as np

from keras.models import *
from keras.layers import *
# from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

from Model import U_Net, U_Net_plus_plus
from Dice import Dice




########## 参数、常量定义部分 ##########
def init_DRIVE():
    global train_dir_root_path, valid_dir_path, valid_label_dir_path, \
        img_size, original_size, model_path, contrast_model_path, \
        total_train_original_img_num, train_img_num, validation_img_num, \
        test_img_num, lr, weight_decay, epochs

    # 数据路径
    train_dir_root_path = './retina_train/'
    valid_dir_path = './retina_train/valid/'
    valid_label_dir_path = './retina_train/valid_label/'

    # 定义图像大小
    img_size = (256, 256)           # 模型输入图像的尺寸
    original_size = (565, 584)      # 视网膜图像的尺寸，方便resize故行列值已经进行了交换，实际尺寸为584x565

    # 定义训练得到的模型保存名
    model_path = './unet.hdf5'
    contrast_model_path = './unetpp.hdf5'

    # 定义数值量
    total_train_original_img_num = 40       # DRIVE总共图片数
    train_img_num = 20                      # 训练集张数，至少大于等于DRIVE的一半，即至少20
    validation_img_num = 10                 # 验证集张数
    test_img_num = 10                       # 测试集张数

    # 定义超参数
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 150


def init_CHASE():
    global train_dir_root_path, valid_dir_path, valid_label_dir_path, \
        img_size, original_size, model_path, contrast_model_path, \
        total_train_original_img_num, train_img_num, validation_img_num, \
        test_img_num, lr, weight_decay, epochs
    # 数据路径
    train_dir_root_path = './retina_train/'
    valid_dir_path = './retina_train/valid/'
    valid_label_dir_path = './retina_train/valid_label/'

    # 定义图像大小
    img_size = (512, 512)  # 模型输入图像的尺寸
    original_size = (960, 999)  # 视网膜图像的尺寸，方便resize故行列值已经进行了交换，实际尺寸为584x565

    # 定义训练得到的模型保存名
    model_path = './unet_chase.hdf5'
    contrast_model_path = './unetpp_chase.hdf5'

    # 定义数值量
    total_train_original_img_num = 28  # CHASE总共图片数
    train_img_num = 20                 # 训练集张数
    validation_img_num = 4             # 验证集张数
    test_img_num = 4                   # 测试集张数

    # 定义超参数
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 150


########## 训练集增强相关 ##########
# 对训练图像进行归一化，对其对应标签进行二值化
def adjust_img(img, mask):
    if np.max(img) > 1:
        img = img / 255
    if np.max(mask) > 1:
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img, mask)


# 定义图像生成器，可以对原始训练集进行扩充
"""
    对下面的函数的解释
    目录结构为：
    retina_train            (即train_path)
        train               (即image_folder)
            image1
            ...
        label               (即label_folder)
            label1
            ...
"""
def train_generator(batch_size, train_path, image_folder, label_folder, aug_dict, image_color_mode="rgb",
                    label_color_mode="grayscale", target_size=(256, 256), seed=1):
    # 定义图像生成器
    image_datagen = ImageDataGenerator(**aug_dict)          # 原图生成器
    label_datagen = ImageDataGenerator(**aug_dict)          # 标签生成器
    image_generator = image_datagen.flow_from_directory(
        train_path,                                         # 这里传的路径应该是训练集数据（包括标签和原始数据）的父目录，
                                                            # 而不是训练原始数据目录，且不含最末尾的'/'
        classes=[image_folder],                             # 这里给定训练集目录名，末尾不含'/'
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)
    label_generator = label_datagen.flow_from_directory(
        train_path,
        classes=[label_folder],
        class_mode=None,
        color_mode=label_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        seed=seed)

    # 结合两个生成器，保证生成的图像是一一对应的
    train_gen = zip(image_generator, label_generator)    # 将训练集和标签组合

    # 对图像进行调整后作为生成图像，成对产出
    for (img, mask) in train_gen:
        img, mask = adjust_img(img, mask)
        yield (img, mask)

# # 不使用data augment的generator
# def train_generator(train_path, train_label_path, num_image=train_img_num, target_size=(256, 256)):
#     train_img_name_list = os.listdir('./retina_train/train')
#     train_label_name_list = os.listdir('./retina_train/label')
#     for i in range(num_image):
#         img = cv2.imread(train_path + train_img_name_list[i], cv2.IMREAD_COLOR)
#         label = cv2.imread(train_label_path + train_label_name_list[i], cv2.IMREAD_GRAYSCALE)
#         # img = io.imread(validation_path + valid_img_name_list[i], as_gray=True)
#         # label = io.imread(valid_label_path + valid_label_name_list[i], as_gray=True)
#
#         img = cv2.resize(img, target_size)
#         # img = np.expand_dims(img, axis=2)
#         img = np.expand_dims(img, axis=0)
#         img = np.array(img, np.uint8)                     # 转化图像格式，由于读入的为float32型，需要转化成uint8，保证输入网络的图像格式正确
#
#         label = cv2.resize(label, target_size)
#         label = np.expand_dims(label, axis=2)
#         label = np.expand_dims(label, axis=0)
#         label = np.array(label, np.uint8)                 # 转化图像格式，由于读入的为float32型，需要转化成uint8，保证输入网络的图像格式正确
#
#         img, label = adjust_img(img, label)     # 对成对的图像进行处理后抛出
#         yield (img, label)





########## 验证集相关 ##########
# 定义验证集生成器
# 验证集生成器实际完成读取验证集图片的工作，成对读取原图与标签
def validation_generator(validation_path, valid_label_path, num_image=10, target_size=(256, 256)):
    valid_img_name_list = os.listdir(valid_dir_path)
    valid_label_name_list = os.listdir(valid_label_dir_path)
    for i in range(num_image):
        img = cv2.imread(validation_path + valid_img_name_list[i], cv2.IMREAD_COLOR)
        label = cv2.imread(valid_label_path + valid_label_name_list[i], cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, target_size)
        # img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        img = np.array(img, np.uint8)                     # 转化图像格式，由于读入的为float32型，需要转化成uint8，保证输入网络的图像格式正确

        label = cv2.resize(label, target_size)
        label = np.expand_dims(label, axis=2)
        label = np.expand_dims(label, axis=0)
        label = np.array(label, np.uint8)                 # 转化图像格式，由于读入的为float32型，需要转化成uint8，保证输入网络的图像格式正确

        img, label = adjust_img(img, label)     # 对成对的图像进行处理后抛出
        yield (img, label)



########## 正式训练部分 ##########
if __name__ == "__main__":
    # 初始化一些变量
    # init_DRIVE()
    init_CHASE()

    # 定义训练集生成器的参数，保证生成的图像包括了旋转、平移等变化
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.05,
                         zoom_range=0.05,
                         horizontal_flip=True,
                         fill_mode='nearest')
    # 定义训练集生成器
    train_gen = train_generator(batch_size=2,
                               train_path=train_dir_root_path[:-1],
                               image_folder='train',
                               label_folder='label',
                               target_size=img_size,
                               aug_dict=data_gen_args)

    # # 不使用data augment的生成器
    # train_gen = train_generator('./retina_train/train',
    #                             './retina_train/label',
    #                             num_image=train_img_num,
    #                             target_size=img_size)

    # 定义模型，默认输入为img_size尺寸
    model = U_Net(input_size=img_size + (3,), lr=lr, wd=weight_decay)
    # 定义模型的保存路径与监督优化的方式
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='val_accuracy',
                                       verbose=1,
                                       save_best_only=True)
    # 定义模型的验证集生成器
    validation_gen = validation_generator(valid_dir_path,
                                          valid_label_dir_path,
                                          num_image=validation_img_num,
                                          target_size=img_size)
    # 训练模型
    model.fit_generator(train_gen,
                       validation_data=validation_gen,
                       validation_steps=1,
                       steps_per_epoch=train_img_num,
                       epochs=epochs,
                       callbacks=[model_checkpoint])

    # # 不使用data augment的训练
    # model.fit_generator(train_gen,
    #                     validation_data=validation_gen,
    #                     validation_steps=1,
    #                     steps_per_epoch=train_img_num,
    #                     epochs=10,
    #                     callbacks=[model_checkpoint])

    print('Finish training. Using lr=', lr)


    # # 训练U-Net++模型
    # model_unetpp = U_Net_plus_plus()
    # contrast_model_checkpoint = ModelCheckpoint(contrast_model_path,
    #                                       monitor='val_accuracy',
    #                                       verbose=1,
    #                                       save_best_only=True)
    # validation_gen = validation_generator(valid_dir_path, valid_label_dir_path)
    # model_unetpp.fit_generator(train_gen,
    #                    validation_data=validation_gen,
    #                    validation_steps=1,
    #                    steps_per_epoch=300,
    #                    epochs=10,
    #                    callbacks=[contrast_model_checkpoint])





