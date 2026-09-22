"""Tests for iridium.world.splats: 3D Gaussian scenes, the reference rasteriser, .ply I/O, and fitting.

Images are kept tiny (32x24) and fits short (<=100 steps) so this file runs in
well under a minute on a single CPU core, per the shared-machine ground rules.
"""

import math
import tempfile
import os

import torch

from iridium.world.camera import Camera
from iridium.world.splats import GaussianScene, render, save_ply, load_ply, fit, _SH_C0

W, H = 32, 24


def make_camera(world_to_camera=None):
    return Camera.from_fov(60.0, W, H, world_to_camera=world_to_camera)


def single_gaussian_scene(mean, sigma, opacity=0.9, rgb=(1.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)):
    means = torch.tensor([mean], dtype=torch.float32)
    log_scales = torch.log(torch.as_tensor(sigma, dtype=torch.float32)).expand(1, 3).clone() \
        if not torch.is_tensor(sigma) or sigma.dim() == 0 else torch.log(sigma).unsqueeze(0)
    quats = torch.tensor([quat], dtype=torch.float32)
    opacity_logits = torch.tensor([math.log(opacity / (1 - opacity))], dtype=torch.float32)
    rgb_t = torch.tensor([rgb], dtype=torch.float32)
    sh0 = (rgb_t - 0.5) / _SH_C0
    return GaussianScene(means, log_scales, quats, opacity_logits, sh0)


def moments(field):
    """Weighted centroid and variance (in pixel units) of a nonnegative [H,W] field."""
    ys = torch.arange(field.shape[0], dtype=torch.float32) + 0.5
    xs = torch.arange(field.shape[1], dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    total = field.sum()
    cu = (gx * field).sum() / total
    cv = (gy * field).sum() / total
    var_u = ((gx - cu) ** 2 * field).sum() / total
    var_v = ((gy - cv) ** 2 * field).sum() / total
    return cu.item(), cv.item(), var_u.item(), var_v.item()


def test_peak_position_and_sigma():
    cam = make_camera()
    z = 3.0
    mean = (0.3, -0.2, z)          # off-axis but small, so cross-terms in J stay ~1%
    sigma = 0.05
    scene = single_gaussian_scene(mean, sigma, opacity=0.9)
    out = render(scene, cam)

    uv, depth = cam.project(scene.means)
    cu, cv, var_u, var_v = moments(out.alpha)
    assert abs(cu - uv[0, 0].item()) < 0.3
    assert abs(cv - uv[0, 1].item()) < 0.3

    expected_sigma_px = math.sqrt((cam.fx * sigma / z) ** 2 + 0.3)
    assert abs(math.sqrt(var_u) - expected_sigma_px) < 0.5
    assert abs(math.sqrt(var_v) - expected_sigma_px) < 0.5

    assert depth[0, 0].item() == z


def test_occlusion_order():
    cam = make_camera(world_to_camera=torch.eye(4))
    near = single_gaussian_scene((0.0, 0.0, 2.0), 0.3, opacity=0.999, rgb=(1.0, 0.0, 0.0))
    far = single_gaussian_scene((0.0, 0.0, 5.0), 0.3, opacity=0.999, rgb=(0.0, 1.0, 0.0))
    scene = near.concat(far)
    out = render(scene, cam)
    cy, cx = H // 2, W // 2
    px = out.rgb[cy, cx]
    assert px[0] > 0.8 and px[1] < 0.2, f"expected red in front, got {px.tolist()}"

    scene_swapped = far.concat(near)  # same geometry, just a different insertion order
    out2 = render(scene_swapped, cam)
    px2 = out2.rgb[cy, cx]
    assert px2[0] > 0.8 and px2[1] < 0.2, "depth sort must not depend on array order"

    # Now actually swap which color is near.
    near_g = single_gaussian_scene((0.0, 0.0, 2.0), 0.3, opacity=0.999, rgb=(0.0, 1.0, 0.0))
    far_r = single_gaussian_scene((0.0, 0.0, 5.0), 0.3, opacity=0.999, rgb=(1.0, 0.0, 0.0))
    out3 = render(near_g.concat(far_r), cam)
    px3 = out3.rgb[cy, cx]
    assert px3[1] > 0.8 and px3[0] < 0.2, f"expected green in front after swap, got {px3.tolist()}"


def test_alpha_background_and_depth():
    cam = make_camera(world_to_camera=torch.eye(4))
    scene = single_gaussian_scene((0.0, 0.0, 4.0), 0.15, opacity=0.7)
    background = (0.2, 0.3, 0.4)
    out = render(scene, cam, background=background)
    assert (out.alpha <= 1.0).all() and (out.alpha >= 0.0).all()

    corner = out.rgb[0, 0]
    assert torch.allclose(corner, torch.tensor(background), atol=1e-4), \
        "far corner should be essentially untouched background"

    cy, cx = H // 2, W // 2
    assert out.alpha[cy, cx] > 0.3
    assert abs(out.depth[cy, cx].item() - 4.0) < 1e-4


def test_covariance_rotation_swaps_extent():
    cam = make_camera(world_to_camera=torch.eye(4))
    mean = (0.0, 0.0, 3.0)
    log_scales = torch.log(torch.tensor([[0.15, 0.02, 0.02]]))

    def build(quat):
        means = torch.tensor([mean])
        quats = torch.tensor([quat], dtype=torch.float32)
        opacity_logits = torch.tensor([4.0])
        sh0 = torch.zeros(1, 3)
        return GaussianScene(means, log_scales.clone(), quats, opacity_logits, sh0)

    out_a = render(build((1.0, 0.0, 0.0, 0.0)), cam)
    _, _, var_u_a, var_v_a = moments(out_a.alpha)
    assert var_u_a > var_v_a * 2, "elongated along camera x should show wider u extent"

    half = math.sqrt(0.5)
    out_b = render(build((half, 0.0, 0.0, half)), cam)  # 90 deg about the view (z) axis
    _, _, var_u_b, var_v_b = moments(out_b.alpha)
    assert var_v_b > var_u_b * 2, "after a 90deg roll the elongation should swap to v"


def test_gradients_are_finite_and_nonzero():
    cam = make_camera()
    scene = GaussianScene.random(6, extent=1.0, generator=torch.Generator().manual_seed(0))
    scene.requires_grad_()
    out = render(scene, cam)
    out.rgb.sum().backward()
    for name in ("means", "log_scales", "quats", "opacity_logits", "sh0"):
        grad = getattr(scene, name).grad
        assert grad is not None, f"{name} has no gradient"
        assert torch.isfinite(grad).all(), f"{name} grad has non-finite entries"
        assert grad.abs().sum() > 0, f"{name} grad is all zero"


def test_fit_reduces_loss():
    gen = torch.Generator().manual_seed(1)
    target = GaussianScene.random(10, extent=0.6, generator=gen)
    cams = [make_camera(world_to_camera=cam.world_to_camera) for cam in
            __import__("iridium.world.camera", fromlist=["orbit"]).orbit(
                make_camera(), target=(0.0, 0.0, 0.0), radius=2.0, n=3)]
    images = [render(target, cam).rgb.detach() for cam in cams]

    noise_gen = torch.Generator().manual_seed(2)
    perturbed = GaussianScene(
        means=target.means + torch.randn(target.means.shape, generator=noise_gen) * 0.05,
        log_scales=target.log_scales + torch.randn(target.log_scales.shape, generator=noise_gen) * 0.1,
        quats=target.quats.clone(),
        opacity_logits=target.opacity_logits + torch.randn(target.opacity_logits.shape, generator=noise_gen) * 0.3,
        sh0=target.sh0 + torch.randn(target.sh0.shape, generator=noise_gen) * 0.3,
    )

    fitted, losses = fit(perturbed, cams, images, steps=80, lr=0.02)
    print(f"fit loss: start={losses[0]:.5f} end={losses[-1]:.5f} ratio={losses[0] / max(losses[-1], 1e-12):.1f}x")
    assert losses[-1] < losses[0] / 3.0, (losses[0], losses[-1])


def test_ply_round_trip_lossless_and_header():
    gen = torch.Generator().manual_seed(3)
    scene = GaussianScene.random(5, extent=0.4, generator=gen)
    scene.sh_rest = torch.randn(5, 3, 3, generator=gen)  # degree 1

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "scene.ply")
        save_ply(scene, path)

        with open(path, "rb") as fh:
            raw = fh.read()
        header_end = raw.find(b"end_header\n")
        header = raw[:header_end].decode("ascii")
        expected_props = (
            ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
            + [f"f_rest_{i}" for i in range(9)]
            + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        )
        found_props = [line.split()[-1] for line in header.splitlines() if line.startswith("property")]
        assert found_props == expected_props

        loaded = load_ply(path)

    assert torch.equal(loaded.means, scene.means)
    assert torch.equal(loaded.sh0, scene.sh0)
    assert torch.allclose(loaded.sh_rest, scene.sh_rest, atol=0)
    assert torch.equal(loaded.opacity_logits, scene.opacity_logits)
    assert torch.equal(loaded.log_scales, scene.log_scales)
    assert torch.equal(loaded.quats, scene.quats)


def test_behind_camera_splats_are_culled():
    cam = make_camera(world_to_camera=torch.eye(4))
    behind = single_gaussian_scene((0.0, 0.0, -5.0), 0.2, opacity=0.99)
    front = single_gaussian_scene((0.0, 0.0, 3.0), 0.2, opacity=0.99, rgb=(0.0, 0.0, 1.0))
    scene = behind.concat(front)
    out = render(scene, cam)
    assert out.visible.tolist() == [False, True]

    only_behind = render(behind, cam)
    assert not only_behind.visible.any()
    assert torch.allclose(only_behind.rgb, torch.zeros(H, W, 3), atol=1e-6)
    assert torch.allclose(only_behind.alpha, torch.zeros(H, W), atol=1e-6)
