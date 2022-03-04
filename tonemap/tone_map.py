import cv2
import numpy as np
from tonemap.bilateral_filter import BilateralFilter


def run_bf_tone_map(image, sigma_space=None, sigma_range=0.4, gamma=0.5, down_sample_factor=1, is_rgb=True):
    '''
    :param image: image is rgb if is_rgb is true
    :param sigma_space:
    :param sigma_range:
    :param gamma:
    :param down_sample_factor:
    :param is_rgb:
    :return:
    '''
    eps = 1e-10

    luminosity_coefficients = np.asarray([20, 40, 1]) / 61.0

    image_ori = image.astype(np.float32)
    if is_rgb:
        image_gray = image_ori[:, :, 0] * luminosity_coefficients[0] + image_ori[:, :, 1] * luminosity_coefficients[1] + image_ori[:, :, 2] * luminosity_coefficients[2]
        image_gray = image_gray / np.max(image_gray)
    else:
        image_gray = image_ori / np.max(image_ori)

    image_intensity_log = np.log10(image_gray + eps)

    # image base and detail scales
    bilateralfilter = BilateralFilter(sigma_space, sigma_range, down_sample_factor, False)
    image_base = bilateralfilter.filter(image_intensity_log)

    # if sigma_space is None:
    #     sigma_space = np.min(image_intensity_log.shape[:2]) * 0.02
    # image_base = cv2.bilateralFilter(image_intensity_log, d=-1, sigmaColor=sigma_range, sigmaSpace=sigma_space)

    image_detail = image_intensity_log - image_base

    image_intensity_bf = np.power(10.0, gamma * image_base + image_detail)

    if is_rgb:
        ch_ratio = image_intensity_bf / (image_gray + eps)
        filt_image_color = image_ori * ch_ratio[:, :, np.newaxis]
        final_image = filt_image_color / np.max(filt_image_color)
    else:
        final_image = image_intensity_bf

    return final_image


if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('/mnt/data/data.set/xintu.data/xt.image.enhancement.540.jpg/rt_tif_16bit_540p/6003_2568.jpg', cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    #img = cv2.imread('/mnt/data/data.set/xintu.data/xt.image.enhancement.540.jpg/rt_tif_16bit_540p/6003_2568.jpg', cv2.IMREAD_ANYDEPTH)
    final_img = run_bf_tone_map(img, is_rgb=True, down_sample_factor=2)

    cv2.imwrite('/home/shengdewu/workspace/ImageAdaptive3DLUT/dataloader/rbf/final.jpg', cv2.cvtColor(np.concatenate([img, (final_img*255.0).astype(np.uint8)], axis=1), cv2.COLOR_RGB2BGR))
    #cv2.imwrite('/home/shengdewu/workspace/ImageAdaptive3DLUT/dataloader/rbf/gray_final.jpg', np.concatenate([img, (final_img * 255.0).astype(np.uint8)], axis=1))
