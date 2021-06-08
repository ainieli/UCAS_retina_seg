"""
    加载训练好的模型，并对视网膜血管进行预测
"""
import numpy as np
import cv2
import os

from keras.models import load_model

from Dice import Dice, DiceLoss


########## 参数、常数定义部分 ##########
def init_DRIVE():
    global test_dir_path, test_label_dir_path, result_dir_path, unetpp_result_dir_path, \
        img_size, original_size, test_img_num, model_path, unetpp_path
    # 路径常数
    test_dir_path = './retina_test/'              # 对视网膜数据的预测路径
    test_label_dir_path = './retina_test_label/'
    result_dir_path = './unet_result/'
    unetpp_result_dir_path = './unetpp_result/'
    try:
        os.mkdir(result_dir_path)
        os.mkdir(unetpp_result_dir_path)
    except:
        pass

    # 图像尺寸
    img_size = (256, 256)                           # 模型输入图像的尺寸
    original_size = (565, 584)                      # 原始图像尺寸

    # 图片张数
    # test_img_num = 10                             # 测试集张数，指定数目
    test_img_num = len(os.listdir(test_dir_path))   # 测试集张数，与目录中实际张数一致

    # 模型名
    model_path = './unet.hdf5'
    unetpp_path = './unetpp.hdf5'


def init_CHASE():
    global test_dir_path, test_label_dir_path, result_dir_path, unetpp_result_dir_path, \
        img_size, original_size, test_img_num, model_path, unetpp_path
    # 路径常数
    test_dir_path = './retina_test_chase/'  # 对视网膜数据的预测路径
    test_label_dir_path = './retina_test_label_chase/'
    result_dir_path = './unet_result_chase/'
    unetpp_result_dir_path = './unetpp_result_chase/'
    try:
        os.mkdir(result_dir_path)
        os.mkdir(unetpp_result_dir_path)
    except:
        pass

    # 图像尺寸
    img_size = (512, 512)  # 模型输入图像的尺寸
    original_size = (960, 999)  # 原始图像尺寸

    # 图片张数
    # test_img_num = 10                             # 测试集张数，指定数目
    test_img_num = len(os.listdir(test_dir_path))  # 测试集张数，与目录中实际张数一致

    # 模型名
    model_path = './unet_chase.hdf5'
    unetpp_path = './unetpp_chase.hdf5'


########## 测试数据相关 ##########
# 定义测试集生成器
# 测试集生成器实际完成读取测试集图片的工作
def test_generator(test_path, num_image, target_size):
    test_img_name_list = os.listdir(test_dir_path)
    for i in range(num_image):
        # img = cv2.imread(test_path + test_img_name_list[i], cv2.IMREAD_COLOR)
        # img = img[:, :, 1]
        img = cv2.imread(test_path + test_img_name_list[i], cv2.IMREAD_GRAYSCALE)
        if np.max(img) > 1:
            img = img / 255
        img = cv2.resize(img, target_size)
        # img = np.expand_dims(img, axis=2)
        img = np.expand_dims(img, axis=0)
        img = np.array(img)  # 转化图像格式，保证输入U-Net网络的图像格式正确
        yield img


# 保存预测得到的图像
def save_result(save_path, results, original_size):
    #print(results.shape)
    test_img_name_list = os.listdir(test_dir_path)
    dice_list = np.zeros(results.shape[0])
    for i, item in enumerate(results):
        img = cv2.resize(item, original_size)
        img = np.array(img * 255, np.uint8)
        _, img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)     # 阈值为255 * 0.25

        y_true = cv2.imread(test_label_dir_path + test_img_name_list[i][:-4] + '.jpg', cv2.IMREAD_GRAYSCALE)
        dice = Dice(y_true, img)
        dice_list[i] = dice
        print('Dice value of image %s is: %.4f' % (test_img_name_list[i][:-4], dice))
        cv2.imwrite(save_path + test_img_name_list[i][:-4] + '_predict.jpg', img)
    print('average Dice value on test set: %.4f' % np.average(dice_list))


########## 加载模型并预测，保存预测结果 ##########
# 保存的预测结果为灰度图像，并非二值图像
if __name__ == "__main__":
    init_DRIVE()
    # init_CHASE()
    MODEL = 'UNET'

    if MODEL == 'UNET':
        model = load_model(model_path, custom_objects={'DiceLoss': DiceLoss})
    elif MODEL == 'UNET++':
        model = load_model(unetpp_path, custom_objects={'DiceLoss': DiceLoss})
    test_gene = test_generator(test_dir_path, test_img_num, img_size)
    results = model.predict_generator(test_gene, test_img_num, verbose=1)
    if MODEL == 'UNET':
        save_result(result_dir_path, results, original_size)
    elif MODEL == 'UNET++':
        save_result(unetpp_result_dir_path, results, original_size)
