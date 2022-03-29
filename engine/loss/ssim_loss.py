import torch
import torch.nn.functional as tnf
from torch.autograd import Variable
import math


class SSIM(torch.nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5, is_multi_scale=True):
        """Initialisation of the DeepLPF loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(SSIM, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size
        self.is_multi_scale = is_multi_scale
        return

    @staticmethod
    def create_window(window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
        可以设定channel参数拓展为3通道

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = SSIM.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(num_channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def gaussian(window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        计算一维的高斯分布向量
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        计算SSIM
        直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
        在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
        正如前面提到的，上面求期望的操作采用高斯核卷积代替。

        ！！！！！ 值越大表示越相似所以损失函数=1-ssim

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        device = img1.device
        (_, num_channel, _, _) = img1.size()

        window = SSIM.create_window(self.ssim_window_size, num_channel)

        window = window.to(device)
        window = window.type_as(img1)

        mu1 = tnf.conv2d(img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = tnf.conv2d(img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = tnf.conv2d(img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = tnf.conv2d(img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = tnf.conv2d(img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
        ssim_map2 = ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        ssim_map = ssim_map1 / ssim_map2

        v1 = 2.0 * sigma12 + c2
        v2 = sigma1_sq + sigma2_sq + c2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs

    def compute_multi_scale_ssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        ！！！！！ 值越大表示越相似所以损失函数=1-ssim

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.shape[2] != img2.shape[2]:
                img1 = img1.transpose(2, 3)

        if img1.shape != img2.shape:
            raise RuntimeError('Input images must have the same shape (%s vs. %s).', img1.shape, img2.shape)
        if img1.ndim != 4:
            raise RuntimeError('Input images must have four dimensions, not %d', img1.ndim)

        device = img1.device

        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        ssims = []
        mcs = []
        for _ in range(levels):
            ssim, cs = self.compute_ssim(img1, img2)

            # Relu normalize (not compliant with original definition)
            ssims.append(ssim)
            mcs.append(cs)

            img1 = tnf.avg_pool2d(img1, (2, 2))
            img2 = tnf.avg_pool2d(img2, (2, 2))

        ssims = torch.stack(ssims)
        mcs = torch.stack(mcs)

        # Simple normalize (not compliant with original definition)
        # TODO: remove support for normalize == True (kept for backward support)
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = ssims ** weights

        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1])
        return output

    def forward(self, img1, img2):
        if self.is_multi_scale:
            return self.compute_multi_scale_ssim(img1, img2)
        else:
            return self.compute_ssim(img1, img2)[0]
