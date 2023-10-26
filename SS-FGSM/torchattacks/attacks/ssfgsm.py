import torch
import torch.nn as nn

from km import kmeans_noise
from slic import sup_noise
from torchattacks.attack import Attack

class SS_FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.SS_FGSM(model, eps=0.007)
        >>> adv_images = attack(cluster_idx, tempsup, images, labels)

    """
    def __init__(self, model, eps=0.007):
        super().__init__("SS_FGSM", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.num_labels = 9

    def forward(self, cluster_idx, tempsup, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)


        noise = torch.randn_like(images).uniform_(-self.eps, self.eps)
        T = 20
        loss = nn.CrossEntropyLoss()

        for i in range(T):
            # Smooth the spatial perturbation
            noise = sup_noise(noise, tempsup)
            noise.requires_grad = True
            adv_images = images + noise
            outputs = self.model(adv_images)
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, noise,
                                       retain_graph=False, create_graph=False)[0]
            noise = noise + self.eps * grad.sign()
            noise = torch.clamp(noise, min=-self.eps, max=self.eps).detach()

            # Smooth the spectral perturbation
            noise = kmeans_noise(noise, cluster_idx)
            noise.requires_grad = True
            adv_images = images + noise
            outputs = self.model(adv_images)
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            grad = torch.autograd.grad(cost, noise,
                                       retain_graph=False, create_graph=False)[0]
            noise = noise + self.eps * grad.sign()
            noise = torch.clamp(noise, min=-self.eps, max=self.eps).detach()
            adv_images = images + noise
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            noise = adv_images - images

        return adv_images
