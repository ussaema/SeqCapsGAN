import torch
import torch.nn as N
import torchvision.models as M
import torchvision.models.resnet as R


class BlockWithLinearOutput(N.Module):
    def __init__(self, block, btype="basic"):
        super(BlockWithLinearOutput, self).__init__()

        assert btype in {"basic", "bottleneck"}

        self.btype = btype
        self.block = block

    def forward(self, x):
        residual = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.btype == "bottleneck":
            out = self.block.relu(out)

            out = self.block.conv3(out)
            out = self.block.bn3(out)

        if self.block.downsample is not None:
            residual = self.block.downsample(x)

        out += residual

        # Disabled
        # out = self.relu(out)

        return out


class ResNetFeatureExtractor(N.Module):
    """Extracts intermediate features in the given ResNet module.
    Given resnet: ResNet and feat_layer, this will perform forward inference
    until the specified layer and return the hidden features.

    Args:
        resnet (ResNet): ResNet module loaded from `torchvision.models`.
        feat_layer (str): Target feature layer specified using Caffe layer names
            proposed in the author's original code. Currently only 'res5c' is
            supported.
    """
    def __init__(self, resnet, feat_layer="res5c"):
        super(ResNetFeatureExtractor, self).__init__()

        assert feat_layer == "res5c", \
            "Current version supports only 'res5c' as the feature layer."

        self.feat_layer = feat_layer
        self.resnet = resnet
        self.layer4_stripped = N.Sequential(*[
            block for i, block in enumerate(resnet.layer4)
            if i < len(resnet.layer4) - 1
        ])
        last_block = self.resnet.layer4[-1]

        if isinstance(last_block, R.BasicBlock):
            last_block_type = "basic"
        elif isinstance(last_block, R.Bottleneck):
            last_block_type = "bottleneck"
        else:
            raise TypeError("Unexpected block type: {}".format(
                type(last_block)
            ))

        self.last_block = BlockWithLinearOutput(
            block=last_block,
            btype=last_block_type
        )

    def forward(self, x):
        """Returns hidden features at feat_layer after performing forward pass
        up to the layer. Spatial features are returned as [B x D x W x H]
        tensors."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.layer4_stripped(x)

        x = self.last_block(x)

        # Disabled
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


if __name__ == "__main__":
    import torch.autograd as A

    # ResNet18 uses BasicBlock as the building block.
    resnet18 = M.resnet18(pretrained=True)

    # ResNet152 uses Bottleneck as the building block.
    resnet50 = M.resnet50(pretrained=True)

    resnet18_fe = ResNetFeatureExtractor(resnet18)
    resnet50_fe = ResNetFeatureExtractor(resnet50)

    x = A.Variable(torch.randn(1, 3, 224, 224))
    h18 = resnet18_fe(x)
    h50 = resnet50_fe(x)

    assert h18.size() == (1, 512, 7, 7)
    assert h50.size() == (1, 2048, 7, 7)