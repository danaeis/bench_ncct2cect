import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        """
        Initialize the DiceLoss module.

        Args:
            weight (Tensor, optional): Optional class weights (not used in this implementation,
                                       but can be extended to support class-balanced dice).
        """
        super(DiceLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        """
        Compute Dice loss between the predicted and target tensors.

        Dice coefficient measures overlap between two binary volumes.
        Dice Loss = 1 - Dice Coefficient

        Args:
            input (Tensor): Predicted tensor of shape [N, C, D, H, W] or [N, C, H, W]
                            (typically after sigmoid or softmax if needed).
            target (Tensor): Ground-truth tensor of the same shape as input.

        Returns:
            loss (Tensor): Scalar loss value (mean Dice loss across batch).
        """
        N = target.size(0)   # Batch size
        C = target.size(1)  # Number of channels / classes
        smooth = 0.0001      # Small constant to avoid division by zero

        # Flatten spatial dimensions: [N, C, D*H*W] or [N, C, H*W]
        input_flat = input.view(N, C, -1)
        target_flat = target.view(N, C, -1)

        # Element-wise multiplication to compute intersection
        intersection = input_flat * target_flat

        # Dice coefficient per sample and class: [N, C]
        dice_score = (2 * intersection.sum(2) + smooth) / (input_flat.sum(2) + target_flat.sum(2) + smooth)
        loss = 1 - dice_score.sum() / N

        return loss