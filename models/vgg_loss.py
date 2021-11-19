import torch
import torchvision


class PerceptualLoss(torch.nn.Module):
    def __init__(self, layer, device='cuda', path=''):
        super(PerceptualLoss, self).__init__()
        self.layer = layer

        self.vgg = torchvision.models.vgg16(pretrained=False)
        map_location = None if device == 'cuda' else 'cpu'
        state_dict = torch.load(path, map_location=map_location)
        self.vgg.load_state_dict(state_dict)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.vgg.eval()
        self.to(device)
        return

    def fetch(self, x, layer):
        fea = None
        for i in range(len(self.vgg.features)):
            x = self.vgg.features[i](x)
            if i == layer:
                fea = x
                break

        return fea

    def forward(self, x, y):

        x_fea = self.fetch(x, self.layer)
        y_fea = self.fetch(y, self.layer)

        return torch.mean((x_fea - y_fea) ** 2)


if __name__ == '__main__':
    device = 'cuda'
    loss = PerceptualLoss(17, device=device, path='/mnt/data/pretrain.model/vgg.model/pytorch/vgg16-397923af.pth')
    x = torch.ones((1, 3, 512, 512), dtype=torch.float32).to(device)
    y = torch.ones((1, 3, 512, 512), dtype=torch.float32).to(device)

    x = loss(x, y)
    print(x)
