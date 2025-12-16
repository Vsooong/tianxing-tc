import numpy as np
import sys
import datetime as dt
from math import ceil

sys.path.append("/data3/WangGuanSong/TianXing/")
from model.tianxing import TianXing
from torch_harmonics import InverseRealSHT
import matplotlib.pyplot as plt
import torch



class Gaussian:
    def __init__(self, noise_amplitude: float = 0.05):
        self.noise_amplitude = noise_amplitude

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x + self.noise_amplitude * torch.randn_like(x)


class SphericalGaussian:
    """
    A mean-zero Gaussian Random Field on the sphere with Matern covariance:
    C = sigma^2 (-Lap + tau^2 I)^(-alpha).

    Lap is the Laplacian on the sphere, I the identity operator,
    and sigma, tau, alpha are scalar parameters.

    Note: C is trace-class on L^2 if and only if alpha > 1.
    """

    def __init__(
        self,
        noise_amplitude: float = 0.05,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
    ):
        self.noise_amplitude = noise_amplitude
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = x.shape
        # Check the ratio
        if 2 * (shape[-2] // 2) != shape[-1] / 2:
            raise ValueError("Lat/lon aspect ration must be N:2N or N+1:2N")

        nlat = 2 * (shape[-2] // 2)  # Noise only support even lat count
        sampler = GaussianRandomFieldS2(
            nlat=nlat,
            alpha=self.alpha,
            tau=self.tau,
            sigma=self.sigma,
            device=x.device,
        )
        sampler = sampler.to(x.device)

        sample_noise = sampler(np.array(shape[:-2]).prod()).reshape(
            *shape[:-2], nlat, 2 * nlat
        )

        # Hack for odd lat coords
        if x.shape[-2] % 2 == 1:
            noise = torch.zeros_like(x)
            noise[..., :-1, :] = sample_noise
            noise[..., -1:, :] = noise[..., -2:-1, :]
        else:
            noise = sample_noise
        if channel_mask is not None:
            noise = noise * channel_mask.to(x.device)
        return x + self.noise_amplitude * noise


class GaussianRandomFieldS2(torch.nn.Module):

    def __init__(
        self,
        nlat: int,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
        radius: float = 1.0,
        grid: str = "equiangular",
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda:0",
    ):
        super().__init__()
        # Number of latitudinal modes.
        self.nlat = nlat

        # Default value of sigma if None is given.
        if alpha < 1.0:
            raise ValueError(f"Alpha must be greater than one, got {alpha}.")

        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - 2.0))

        # Inverse SHT
        self.isht = (
            InverseRealSHT(self.nlat, 2 * self.nlat, grid=grid, norm="backward")
            .to(dtype=dtype)
            .to(device=device)
        )

        # Square root of the eigenvalues of C.
        sqrt_eig = (
            torch.tensor([j * (j + 1) for j in range(self.nlat)], device=device)
            .view(self.nlat, 1)
            .repeat(1, self.nlat + 1)
        )
        sqrt_eig = torch.tril(
            sigma * (((sqrt_eig / radius**2) + tau**2) ** (-alpha / 2.0))
        )
        sqrt_eig[0, 0] = 0.0
        sqrt_eig = sqrt_eig.unsqueeze(0)
        self.register_buffer("sqrt_eig", sqrt_eig)

        # Save mean and var of the standard Gaussian.
        # Need these to re-initialize distribution on a new device.
        mean = torch.tensor([0.0], device=device).to(dtype=dtype)
        var = torch.tensor([1.0], device=device).to(dtype=dtype)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)

    def forward(self, N: int, xi: torch.Tensor | None = None) -> torch.Tensor:
        # Sample Gaussian noise.
        if xi is None:
            gaussian_noise = torch.distributions.normal.Normal(self.mean, self.var)
            xi = gaussian_noise.sample(
                torch.Size((N, self.nlat, self.nlat + 1, 2))
            ).squeeze()
            xi = torch.view_as_complex(xi)

        # Karhunen-Loeve expansion.
        u = self.isht(xi * self.sqrt_eig)

        return u


class Brown:
    def __init__(self, noise_amplitude: float = 0.05, reddening: int = 2):
        self.reddening = reddening
        self.noise_amplitude = noise_amplitude

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        """
        shape = x.shape
        noise = self._generate_noise_correlated(tuple(shape), device=x.device)

        return x + self.noise_amplitude * noise

    def _generate_noise_correlated(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        """Utility class for producing brown noise."""
        noise = torch.randn(*shape, device=device)
        x_white = torch.fft.rfft2(noise)
        S = (
            torch.abs(torch.fft.fftfreq(shape[-2], device=device).reshape(-1, 1))
            ** self.reddening
            + torch.fft.rfftfreq(shape[-1], device=device) ** self.reddening
        )
        S = 1 / S
        S[..., 0, 0] = 0
        S = S / torch.sqrt(torch.mean(S**2))

        x_shaped = x_white * S
        noise_shaped = torch.fft.irfft2(x_shaped, s=shape[-2:])
        return noise_shaped


class BredVector:
    """
    model : Callable[[torch.Tensor], torch.Tensor]
        Dynamical model, typically this is the prognostic AI model.
        TODO: Update to prognostic looper
    noise_amplitude : float, optional
        Noise amplitude, by default 0.05
    integration_steps : int, optional
        Number of integration steps to use in forward call, by default 20
    ensemble_perturb : bool, optional
        Perturb the ensemble in an interacting fashion, by default False
    seeding_perturbation_method : Perturbation, optional
        Method to seed the Bred Vector perturbation, by default Brown Noise

    """

    def __init__(
        self,
        model: TianXing,
        noise_amplitude: float = 0.05,
        integration_steps: int = 20,
        ensemble_perturb: bool = False,
        # seeding_perturbation_method=Brown(),
        seeding_perturbation_method=SphericalGaussian(
            noise_amplitude=0.05, alpha=2, tau=20.0, sigma=None
        ),
    ):
        self.model = model
        self.noise_amplitude = noise_amplitude
        self.ensemble_perturb = ensemble_perturb
        self.integration_steps = integration_steps
        self.seeding_perturbation_method = seeding_perturbation_method
        latitude = torch.from_numpy(np.linspace(90, -90, 721)).float().to(model.device)
        wlat = torch.cos(torch.deg2rad(latitude))
        wlat /= wlat.mean()
        self.wlat = wlat.unsqueeze(-1).unsqueeze(0).unsqueeze(0)

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dx = self.seeding_perturbation_method(x)
        dx -= x

        xd = torch.clone(x)
        xd = self.model.forward(xd, times=4)
        # Run forward model
        for k in range(self.integration_steps):
            x1 = x + dx
            x2 = self.model.forward(x1, times=4)
            if self.ensemble_perturb:
                dx1 = x2 - xd
                dx = dx1 + self.noise_amplitude * (dx - dx.mean(dim=0))
            else:
                dx = x2 - xd

        # dx = dx * self.wlat

        gamma = torch.norm(x) / torch.norm(x + dx)

        return x + dx * self.noise_amplitude * gamma


def load_input(model_wrapper, data_time=dt.datetime(2021, 7, 19, 18, 0)):
    assert isinstance(model_wrapper, TianXing)
    all_fields, target_fields, timestamp, latitude = (
        model_wrapper.load_input_from_arxiv(data_time, return_all=True)
    )
    all_fields = model_wrapper.normalise(all_fields)
    input_field = all_fields.to(model_wrapper.device, dtype=torch.float32)
    latitude = latitude.to(model_wrapper.device, dtype=torch.float32)
    return input_field, latitude


if __name__ == "__main__":
    import gc

    per_name = "BredVector"
    model_path = "/data3/WangGuanSong/Weaformer/all_models/weaformer_v2.0/"
    lead_time = 24
    model_wrapper = TianXing(root_path=model_path, lead_time=lead_time)
    input_field, latitude = load_input(model_wrapper)
    print(input_field.max(), input_field.min(), input_field.mean(), input_field.std())
    print(input_field.shape)
    exit(0)
    noise_amplitude = 0.05
    perturbation_dict = {
        "Gaussian": Gaussian(noise_amplitude=noise_amplitude),
        "SphericalGaussian": SphericalGaussian(
            noise_amplitude=noise_amplitude, alpha=2, tau=20.0, sigma=None
        ),
        "Brown": Brown(noise_amplitude=noise_amplitude, reddening=1.0),
        "BredVector": BredVector(
            model=model_wrapper,
            noise_amplitude=noise_amplitude,
            integration_steps=25,
            ensemble_perturb=False,
            seeding_perturbation_method=Brown(
                noise_amplitude=noise_amplitude, reddening=1.0
            ),
        ),
    }

    perturbation = perturbation_dict[per_name]
    nensemble = 51
    batch_size = 2
    if batch_size is None:
        batch_size = nensemble
    number_of_batches = ceil(nensemble / batch_size)
    ensemble_index = 0
    all_perts = []
    for i in range(number_of_batches):
        ensemble_index = i * batch_size
        mini_batch_size = min(batch_size, nensemble - ensemble_index)
        batch_input = input_field.repeat(mini_batch_size, 1, 1, 1)
        perturbed_field = perturbation(batch_input)
        perts = (perturbed_field - batch_input).cpu()
        perts = perts * 278.845 / torch.mean(torch.norm(perts, p=2, dim=(1, 2, 3)))
        all_perts.append(perts)
        gc.collect()

    # add 0 perturbation for control
    perts = torch.zeros_like(input_field).cpu()
    all_perts.append(perts)
    perts = torch.cat(all_perts, dim=0)
    print(perts.shape)
    print(
        perts.min(),
        perts.max(),
        perts.mean(),
        perts.std(),
        torch.norm(perts, p=2, dim=(1, 2, 3)),
    )
    # save the perturbation
    torch.save(
        perts, f"/data3/WangGuanSong/TianXing/Perturbation samples/{per_name}.pt"
    )