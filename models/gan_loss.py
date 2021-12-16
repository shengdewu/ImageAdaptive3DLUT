import torch


class GanLoss(torch.nn.Module):
    def __init__(self, loss_model, device='cuda'):
        super(GanLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(1.0), persistent=False)
        self.register_buffer('fake_label', torch.tensor(0.0), persistent=False)
        self.loss_model = loss_model
        self.to(device)
        return

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, x, target_is_real: bool):
        target_tensor = self.get_target_tensor(x, target_is_real)
        return self.loss_model(x, target_tensor)
