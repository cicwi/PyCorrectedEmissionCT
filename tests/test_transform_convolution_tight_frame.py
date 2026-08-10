import pytest
import numpy as np
from corrct.operators import TransformConvolutionTightFrame, __has_torch__


@pytest.fixture(params=[3, 4, 5])
def kernels(request):
    kernel_size = request.param
    num_kernels = kernel_size**2

    # Generate random kernels
    kernels = np.random.randn(num_kernels, kernel_size, kernel_size)

    # Orthogonalize the kernels using QR decomposition
    def orthogonalize(kernels):
        kernels_reshaped = kernels.reshape(kernels.shape[0], -1)
        Q, _ = np.linalg.qr(kernels_reshaped)
        return Q.reshape(kernels.shape)

    # Orthogonalize and exclude the first kernel (0 frequency)
    kernels = orthogonalize(kernels)[1:]

    return {f'{kernel_size}x{kernel_size}': kernels}


@pytest.fixture(params=[32, 128, 1024])
def images(request):
    image_size = request.param
    return {f'{image_size}x{image_size}': np.random.randn(image_size, image_size)}


def test_backend_direct_consistency(kernels, images):
    kernel_name, kernel = next(iter(kernels.items()))
    image_name, image = next(iter(images.items()))

    if not __has_torch__:
        pytest.skip("PyTorch is not available")

    # Create operators for each backend
    op_torch = TransformConvolutionTightFrame(image.shape, kernel, backend='torch')
    op_scipy = TransformConvolutionTightFrame(image.shape, kernel, backend='scipy')
    # op_fft = TransformConvolutionTightFrame(image.shape, kernel, backend='fft')

    # Compute direct and adjoint for each backend
    y_torch = op_torch(image)
    y_scipy = op_scipy(image)
    # y_fft = op_fft(image)

    # Check consistency between backends
    assert np.allclose(
        y_torch, y_scipy, rtol=1e-5, atol=1e-5
    ), f"Direct operation inconsistent for {kernel_name} kernels and {image_name} image"
    # assert np.allclose(
    #     y_torch, y_fft, rtol=1e-5, atol=1e-5
    # ), f"Direct operation inconsistent for {kernel_name} kernels and {image_name} image"


def test_backend_adjoint_consistency(kernels, images):
    kernel_name, kernel = next(iter(kernels.items()))
    image_name, image = next(iter(images.items()))

    if not __has_torch__:
        pytest.skip("PyTorch is not available")

    # Create operators for each backend
    op_torch = TransformConvolutionTightFrame(image.shape, kernel, backend='torch')
    op_scipy = TransformConvolutionTightFrame(image.shape, kernel, backend='scipy')
    # op_fft = TransformConvolutionTightFrame(image.shape, kernel, backend='fft')

    y = op_torch(image)

    x_torch = op_torch.T(y)
    x_scipy = op_scipy.T(y)
    # x_fft = op_fft.T(y)

    # print(f"{kernel_name = }: {x_torch.shape = }, {x_scipy.shape = }")
    # print(f"{x_torch = }")
    # print(f"{x_scipy = }")

    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(x_torch)
    # axs[0, 1].imshow(x_scipy)
    # axs[1, 0].imshow(image - x_torch / kernel.shape[0])
    # axs[1, 1].imshow(image - x_scipy / kernel.shape[0])
    # fig.tight_layout()
    # plt.show()

    # Check consistency between backends
    assert np.allclose(
        x_torch[:-1, :-1], x_scipy[:-1, :-1], rtol=1e-5, atol=1e-5
    ), f"Adjoint operation inconsistent for {kernel_name} kernels and {image_name} image"
    # assert np.allclose(
    #     x_torch, x_fft, rtol=1e-5, atol=1e-5
    # ), f"Adjoint operation inconsistent for {kernel_name} kernels and {image_name} image"


@pytest.mark.benchmark(group="convolution-performance")
@pytest.mark.parametrize("backend", ['torch', 'scipy'])  # , 'fft'
def test_performance_benchmark_direct(request: pytest.FixtureRequest, kernels, images, backend):
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    if not __has_torch__ and backend == "torch":
        pytest.skip("PyTorch is not available")

    kernel_name, kernel = next(iter(kernels.items()))
    image_name, image = next(iter(images.items()))

    # Create operators for each backend
    op = TransformConvolutionTightFrame(image.shape, kernel, backend=backend)

    # Benchmark direct operation
    benchmark.group = f"Direct operation - {kernel_name} kernels - {image_name} image"
    benchmark(op, image)


@pytest.mark.benchmark(group="convolution-performance")
@pytest.mark.parametrize("backend", ['torch', 'scipy'])  # , 'fft'
def test_performance_benchmark_adjoint(request: pytest.FixtureRequest, kernels, images, backend):
    try:
        benchmark = request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("benchmark fixture not available")

    if not __has_torch__ and backend == "torch":
        pytest.skip("PyTorch is not available")

    kernel_name, kernel = next(iter(kernels.items()))
    image_name, image = next(iter(images.items()))

    # Create operators for each backend
    op = TransformConvolutionTightFrame(image.shape, kernel, backend=backend)

    # Benchmark adjoint operation
    y = op(image)

    benchmark.group = f"Adjoint operation - {kernel_name} kernels - {image_name} image"
    benchmark(op.T, y)
