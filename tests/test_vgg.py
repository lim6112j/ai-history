"""Phase 4 tests: VGG shapes, make_layers, and the 3×3-stack receptive field."""

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_history.metrics import GradientHistory
from dl_history.models.vgg import VGG11, VGG16, VGG11Wide, make_layers
from dl_history.train import train_model

# Trunk (3×3, 8 convs) + classifier fc1/fc2 512² + fc3
VGG11_PARAM_COUNT = 9_749_770
VGG16_PARAM_COUNT = 15_243_978
VGG11_WIDE_PARAM_COUNT = 11_387_914
# BatchNorm adds weight+bias per channel: 2 × (64+128+256+256+512·4)
VGG11_BN_PARAM_COUNT = 9_755_274

ALL_VARIANTS = [VGG11, VGG16, VGG11Wide]


@pytest.mark.parametrize("build", ALL_VARIANTS)
@pytest.mark.parametrize("batch_norm", [False, True])
def test_forward_output_shape(build, batch_norm: bool) -> None:
    model = build(batch_norm=batch_norm)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.parametrize(
    "build,expected",
    [
        (VGG11, VGG11_PARAM_COUNT),
        (VGG16, VGG16_PARAM_COUNT),
        (VGG11Wide, VGG11_WIDE_PARAM_COUNT),
    ],
)
def test_parameter_counts(build, expected: int) -> None:
    assert sum(p.numel() for p in build().parameters()) == expected


def test_batch_norm_parameter_count() -> None:
    assert sum(p.numel() for p in VGG11(batch_norm=True).parameters()) == VGG11_BN_PARAM_COUNT


def test_input_padding_makes_five_pools_end_at_one_by_one() -> None:
    """28×28 + 2·2 padding = 32×32 → five stride-2 pools → 1×1."""
    trunk = VGG11().features
    padded = nn.ZeroPad2d(2)(torch.randn(2, 1, 28, 28))
    assert trunk(padded).shape == (2, 512, 1, 1)


@pytest.mark.parametrize("batch_norm,expected", [(False, False), (True, True)])
def test_batch_norm_modules_present_only_when_requested(batch_norm: bool, expected: bool) -> None:
    model = VGG11(batch_norm=batch_norm)
    has_bn = any(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    assert has_bn is expected


def test_two_3x3_convs_have_same_receptive_field_as_one_5x5() -> None:
    """The VGG argument, verified on the tensors themselves.

    All-ones weights, no bias, no ReLU → both trunks are linear. A single
    non-zero input pixel must light up the same 5×5 output region in each.
    """
    # Arrange: identical weights, one impulse in the middle of a 9×9 input
    stack = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.Conv2d(1, 1, 3, padding=1))
    wide = nn.Conv2d(1, 1, 5, padding=2)
    for module in (stack[0], stack[1], wide):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    impulse = torch.zeros(1, 1, 9, 9)
    impulse[0, 0, 4, 4] = 1.0

    # Act
    stack_out = stack(impulse)[0, 0]
    wide_out = wide(impulse)[0, 0]

    # Assert: both respond on exactly the same 5×5 patch centred on the impulse
    stack_support = stack_out != 0
    wide_support = wide_out != 0
    assert torch.equal(stack_support, wide_support)
    assert stack_support.sum().item() == 25
    assert stack_support[2:7, 2:7].all()
    assert not stack_support[0, :].any() and not stack_support[:, 0].any()


def test_stacked_3x3_uses_fewer_weights_than_one_5x5() -> None:
    """18·C² weights (two 3×3) vs 25·C² (one 5×5) for the same channel pair."""
    channels = 16
    stacked = make_layers([channels, channels], in_channels=channels, kernel_size=3)
    wide = make_layers([channels], in_channels=channels, kernel_size=5)

    stacked_weights = sum(p.numel() for p in stacked.parameters())
    wide_weights = sum(p.numel() for p in wide.parameters())

    assert stacked_weights == 2 * (3 * 3 * channels * channels + channels)
    assert wide_weights == 5 * 5 * channels * channels + channels
    assert stacked_weights < wide_weights
    assert sum(p.numel() for p in VGG11().parameters()) < sum(
        p.numel() for p in VGG11Wide().parameters()
    )


@pytest.mark.parametrize(
    "build,conv_layers",
    [(VGG11, 8), (VGG16, 13), (VGG11Wide, 5)],
)
def test_weight_layer_names_cover_trunk_and_classifier(build, conv_layers: int) -> None:
    names = build().weight_layer_names()
    assert len(names) == conv_layers + 3
    assert names[-3:] == ["fc1.weight", "fc2.weight", "fc3.weight"]


def test_vgg16_is_deeper_than_vgg11() -> None:
    assert len(VGG16().weight_layer_names()) > len(VGG11().weight_layer_names())


def test_unknown_config_entry_rejected() -> None:
    with pytest.raises(TypeError):
        make_layers([64, "X"])


@pytest.mark.parametrize("dropout", [-0.1, 1.0])
def test_out_of_range_dropout_rejected(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        VGG11(dropout=dropout)


def test_unknown_init_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown init"):
        VGG11(init="xavier")


def test_he_init_replaces_default_conv_weights() -> None:
    torch.manual_seed(0)
    default = VGG11()
    torch.manual_seed(0)
    he = VGG11(init="he")
    assert not torch.equal(default.features[0].weight, he.features[0].weight)


def test_he_init_leaves_batch_norm_parameters_alone() -> None:
    model = VGG11(batch_norm=True, init="he")
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert torch.all(module.weight == 1) and torch.all(module.bias == 0)


def _trunk_output_std(init: str) -> float:
    """Std of the final trunk activation — how much signal survives 8 convs."""
    torch.manual_seed(0)
    model = VGG11(init=init).eval()
    with torch.no_grad():
        deep = model.pad(torch.randn(8, 1, 28, 28))
        for layer in model.features:
            deep = layer(deep)
    return deep.std().item()


def test_he_init_keeps_more_signal_alive_than_default_init() -> None:
    """Why the no-BatchNorm net needs He init: default init collapses the signal."""
    assert _trunk_output_std("he") > _trunk_output_std("default")


def test_train_model_records_gradient_norms() -> None:
    # Arrange: tiny loader (3 batches of 4) + one training epoch
    images = torch.randn(12, 1, 28, 28)
    labels = torch.randint(0, 10, (12,))
    loaders = (
        DataLoader(TensorDataset(images, labels), batch_size=4),
        DataLoader(TensorDataset(images, labels), batch_size=4),
    )
    grad_history = GradientHistory()

    # Act
    train_model(VGG11(dropout=0.0), loaders, epochs=1, lr=0.01, grad_history=grad_history)

    # Assert: one record per training step, one norm per weight layer
    assert len(grad_history.grad_norms) == 3
    assert all(len(row) == 11 for row in grad_history.grad_norms)
