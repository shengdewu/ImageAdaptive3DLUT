import numpy as np
import cv2


def log(x):
    log_base = 1.0 / np.log(10.0)
    return np.log(x) * log_base


def exp(x):
    return np.power(10, x)


def hdr(img, intensity=None, down_scaler=1.0, unnormalizing_value=255.0):

    img_rgb = img.astype(np.float32)

    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    if intensity is None:
        intensity = (r * 20 + g * 40 + b + 1.0) / 61.0
        intensity = intensity / np.max(intensity)

    log_intensity = log(intensity)

    space_sigma = 0.02 * min(img_rgb.shape[0], img_rgb.shape[1])
    range_sigma = 0.4
    color_sigma = 0.02 * np.min(log_intensity)

    filter_log_intensity = cv2.bilateralFilter(log_intensity, d=int(space_sigma), sigmaColor=color_sigma, sigmaSpace=space_sigma)

    #filter_log_intensity = guided_filter(log_intensity, log_intensity, r=5, eps=0.05, s=down_scaler)

    detail = log_intensity - filter_log_intensity

    max_value = np.max(filter_log_intensity)
    min_value = np.min(filter_log_intensity)
    gamma = log(100) / (max_value - min_value)

    new_intensity = exp(filter_log_intensity * gamma + detail)

    ratio = new_intensity / intensity

    img_rgb[:, :, 0] *= ratio
    img_rgb[:, :, 1] *= ratio
    img_rgb[:, :, 2] *= ratio
    img_rgb = img_rgb / np.max(img_rgb)
    # scale_factor = 1.0 / exp(max_value * gamma)
    #
    # img_rgb[:, :, 0] = np.clip(0, unnormalizing_value, unnormalizing_value*np.power(scale_factor * img_rgb[:, :, 0], 1.0/2.2))
    # img_rgb[:, :, 1] = np.clip(0, unnormalizing_value, unnormalizing_value*np.power(scale_factor * img_rgb[:, :, 1], 1.0/2.2))
    # img_rgb[:, :, 2] = np.clip(0, unnormalizing_value, unnormalizing_value*np.power(scale_factor * img_rgb[:, :, 2], 1.0/2.2))

    return img_rgb

if __name__ == '__main__':
    img = cv2.cvtColor(cv2.imread('/mnt/data/data.set/xintu.data/xt.image.enhancement.540.jpg/rt_tif_16bit_540p/6003_2568.jpg', cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
    #img = cv2.imread('/mnt/data/data.set/xintu.data/xt.image.enhancement.540.jpg/rt_tif_16bit_540p/6003_2568.jpg', cv2.IMREAD_ANYDEPTH)
    final_img = hdr(img)

    cv2.imwrite('/home/shengdewu/workspace/ImageAdaptive3DLUT/dataloader/rbf/final_hdr.jpg', cv2.cvtColor(np.concatenate([img, (final_img*255.0).astype(np.uint8)], axis=1), cv2.COLOR_RGB2BGR))
    #cv2.imwrite('/home/shengdewu/workspace/ImageAdaptive3DLUT/dataloader/rbf/gray_final.jpg', np.concatenate([img, (final_img * 255.0).astype(np.uint8)], axis=1))
