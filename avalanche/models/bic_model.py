import torch


class BiasLayer(torch.nn.Module):
    """Bias layers with alpha and beta parameters
    
    Bias layers used in Bias Correction (BiC) plugin.
    "Wu, Yue, et al. "Large scale incremental learning." Proceedings 
    of the IEEE/CVF Conference on Computer Vision and Pattern 
    Recognition. 2019"
    """

    def __init__(self, device, clss):
        """
        :param device: device used by the main model. 'cpu' or 'cuda'
        :param clss: list of classes of the current layer. This are use 
            to identify the columns which are multiplied by the Bias 
            correction Layer.
        """
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(1, device=device))

        self.clss = torch.Tensor(list(clss)).long().to(device)
        self.not_clss = None

    def forward(self, x):
        # x = x[0]
        if len(x.shape) == 3:
            x = x[0]
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        alpha = torch.ones_like(x)
        beta = torch.ones_like(x)

        alpha[:, self.clss] = self.alpha
        beta[:, self.clss] = self.beta
        # print(alpha, beta)
        return alpha * x + beta
