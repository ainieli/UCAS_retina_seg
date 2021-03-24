"""
    对DRIVE数据集进行预处理，主要完成：
    1.提取训练集、验证集，修改数据格式
    2.将原始数据复制到指定目录下
"""
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


########## 参数、常量定义部分 ##########
def init_DRIVE():
    global train_original_dir_path, train_original_label_dir_path, test_original_dir_path, \
        test_original_label_dir_path, train_dir_path, label_dir_path, test_dir_path, \
        test_label_dir_path, valid_dir_path, valid_label_dir_path, total_train_original_img_num, \
        train_img_num, validation_img_num, test_img_num

    # 原始数据集路径定义
    train_original_dir_path = './RAW_DATA/DRIVE/training/images/'            # DRIVE训练集路径
    train_original_label_dir_path = './RAW_DATA/DRIVE/training/1st_manual/'  # DRIVE训练集对应标签路径
    test_original_dir_path = './RAW_DATA/DRIVE/test/images/'                 # DRIVE测试集路径，可取该路径下部分图像做训练集
    test_original_label_dir_path = './RAW_DATA/DRIVE/test/1st_manual/'       # 当DRIVE测试集部分图像作为训练数据时，对应的标签路径

    # 经过处理的数据路径
    train_dir_path = './retina_train/train/'
    label_dir_path = './retina_train/label/'
    test_dir_path = './retina_test/'
    test_label_dir_path = './retina_test_label/'
    valid_dir_path = './retina_train/valid/'
    valid_label_dir_path = './retina_train/valid_label/'

    # 定义图像大小
    # img_size = (256, 256)           # 模型输入图像的尺寸

    # 定义数值量，不可随意定义，受DRIVE数据集总张数的约束
    # 目前要求下面三者之和等于total_train_original_img_num，否则不执行后续
    total_train_original_img_num = 40       # DRIVE总共图片数
    train_img_num = 20                      # 训练集张数，至少大于等于DRIVE的一半，即至少20
    validation_img_num = 10                 # 验证集张数
    test_img_num = 10                       # 测试集张数


def init_CHASE():
    global train_original_dir_path, train_original_label_dir_path, test_original_dir_path, \
        test_original_label_dir_path, train_dir_path, label_dir_path, test_dir_path, \
        test_label_dir_path, valid_dir_path, valid_label_dir_path, total_train_original_img_num, \
        train_img_num, validation_img_num, test_img_num

    # 原始数据集路径定义
    train_original_dir_path = './RAW_DATA/CHASE/training/images/'  # DRIVE训练集路径
    train_original_label_dir_path = './RAW_DATA/CHASE/training/1st_manual/'  # DRIVE训练集对应标签路径
    test_original_dir_path = './RAW_DATA/CHASE/test/images/'  # DRIVE测试集路径，可取该路径下部分图像做训练集
    test_original_label_dir_path = './RAW_DATA/CHASE/test/1st_manual/'  # 当DRIVE测试集部分图像作为训练数据时，对应的标签路径

    # 经过处理的数据路径
    train_dir_path = './retina_train_chase/train/'
    label_dir_path = './retina_train_chase/label/'
    test_dir_path = './retina_test_chase/'
    test_label_dir_path = './retina_test_label_chase/'
    valid_dir_path = './retina_train_chase/valid/'
    valid_label_dir_path = './retina_train_chase/valid_label/'

    # 定义图像大小
    # img_size = (256, 256)           # 模型输入图像的尺寸

    # 定义数值量，不可随意定义，受DRIVE数据集总张数的约束
    # 目前要求下面三者之和等于total_train_original_img_num，否则不执行后续
    total_train_original_img_num = 28  # DRIVE总共图片数
    train_img_num = 20  # 训练集张数，至少大于等于DRIVE的一半，即至少20
    validation_img_num = 4  # 验证集张数
    test_img_num = 4  # 测试集张数


def clear_file(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            os.remove(os.path.join(root, name))



########## 数据预处理部分，实现图像G通道提取、更名、训练集/验证集的移动 ##########
if __name__ == "__main__":

    # 清除源文件
    # init_DRIVE()
    # clear_file("./retina_train")
    init_CHASE()
    clear_file("./retina_train_chase")

    clear_file(test_dir_path)
    clear_file(test_label_dir_path)

    # 对DRIVE训练集、验证集和测试集张数分配合理性的检验
    if train_img_num + validation_img_num + test_img_num != total_train_original_img_num:
        print("不合理的图像张数分配！")
        exit(0)

    # 读取 ./DRIVE/training/images/ 文件夹中的视网膜数据
    # 默认该目录下所有图片都是训练集
    for img_name in os.listdir(train_original_dir_path):
        img = plt.imread(train_original_dir_path + img_name)          # 此处获取RGB图像，通道对应axis=2
        # img = cv2.resize(img, img_size)
        if img_name[0:2] == 'Im':
            plt.imsave(train_dir_path + img_name[6:9] + '.jpg', img)
        else:
            plt.imsave(train_dir_path + img_name[0:2] + '.jpg', img)
            # cv2.imwrite(train_dir_path + img_name[0:2] + '.jpg', img)
    # 读取对应的图像标签
    # 默认该目录下所有标签都是训练集
    for img_name in os.listdir(train_original_label_dir_path):
        img = plt.imread(train_original_label_dir_path + img_name)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        # img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=2)
        if img_name[0:2] == 'Im':
            cv2.imwrite(label_dir_path + img_name[6:9] + '.jpg', img * 255)
        else:
            # plt.imsave(label_dir_path + img_name[0:2] + '.jpg', img)
            cv2.imwrite(label_dir_path + img_name[0:2] + '.jpg', img)     # 保存图像并转化格式（.gif->.jpg）

    # 读取部分训练集/测试集/验证集数据
    # 根据validation_img_num、train_img_num、test_img_num决定该目录下图像的分配
    for i, img_name in enumerate(sorted(os.listdir(test_original_dir_path))):
        img = plt.imread(test_original_dir_path + img_name)
        # img = cv2.resize(img, img_size)
        if i < test_img_num:
            if img_name[0:2] == 'Im':
                plt.imsave(test_dir_path + img_name[6:9] + '.jpg', img)
            else:
                plt.imsave(test_dir_path + img_name[0:2] + '.jpg', img)
                # cv2.imwrite(test_dir_path + img_name[0:2] + '.jpg', img)       # 前几张测试用，保存图像并转化格式（.tif->.jpg）
        elif i >= test_img_num and i < test_img_num + validation_img_num:
            if img_name[0:2] == 'Im':
                plt.imsave(valid_dir_path + img_name[6:9] + '.jpg', img)
            else:
                plt.imsave(valid_dir_path + img_name[0:2] + '.jpg', img)
                # cv2.imwrite(valid_dir_path + img_name[0:2] + '.jpg', img)      # 中间几张训练时验证用，保存图像并转化格式（.tif->.jpg）
        else:                                                                # 从 DRIVE/test/images/ 文件夹中分部分数据到训练集
            if img_name[0:2] == 'Im':
                plt.imsave(valid_dir_path + img_name[6:9] + '.jpg', img)
            else:
                plt.imsave(valid_dir_path + img_name[0:2] + '.jpg', img)
                # cv2.imwrite(train_dir_path + img_name[0:2] + '.jpg', img)      # 后几张图像训练时用，保存图像并转化格式（.tif->.jpg）

    # 读取验证集的图像标签
    for i, img_name in enumerate(sorted(os.listdir(test_original_label_dir_path))):
        img = plt.imread(test_original_label_dir_path + img_name)
        #cv2.imshow('img',img)
        #cv2.waitKey(0)
        # img = cv2.resize(img, img_size)
        img = np.expand_dims(img, axis=2)
        if i >= test_img_num and i < test_img_num + validation_img_num:      # 中间几张验证时用
            if img_name[0:2] == 'Im':
                cv2.imwrite(valid_label_dir_path + img_name[6:9] + '.jpg', img * 255)
            else:
                # plt.imsave(valid_label_dir_path + img_name[0:2] + '.jpg', img)
                cv2.imwrite(valid_label_dir_path + img_name[0:2] + '.jpg', img)  # 保存图像并转化格式（.gif->.jpg）
        elif i >= test_img_num + validation_img_num:                         # 后面几张训练时用
            if img_name[0:2] == 'Im':
                cv2.imwrite(label_dir_path + img_name[6:9] + '.jpg', img * 255)
            else:
                # plt.imsave(label_dir_path + img_name[0:2] + '.jpg', img)
                cv2.imwrite(label_dir_path + img_name[0:2] + '.jpg', img)        # 保存图像并转化格式（.gif->.jpg）
        elif i < test_img_num:
            if img_name[0:2] == 'Im':
                cv2.imwrite(test_label_dir_path + img_name[6:9] + '.jpg', img * 255)
            else:
                cv2.imwrite(test_label_dir_path + img_name[0:2] + '.jpg', img)

    # cv2.destroyAllWindows()