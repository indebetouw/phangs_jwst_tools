import os
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import webbpsf
from astropy import wcs
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table, vstack
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from photutils.detection import IRAFStarFinder
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import map_coordinates
from statsmodels.formula.api import logit


def fake_image(
    test_hdu, test_psf, n_sources, charflux=100 * u.uJy, return_hdu=False, fluxspread=2
):
    """
    Generate a fake image with random sources.

    Parameters:
    test_hdu (HDUList): Input HDU list.
    test_psf (HDUList): Point Spread Function HDU list.
    n_sources (int): Number of sources to generate. Note only sources
                     inside the valid regions of the image will be generated.
    charflux (Quantity, optional): Characteristic flux of the sources. Defaults to 100 uJy.
    return_hdu (bool, optional): If True, return an HDU list instead of an image.
    fluxspread (int, optional): Spread of the log normal flux distribution. Defaults to 2.

    Returns:
    ndarray or HDUList: Fake image with random sources or HDU list if return_hdu is True.
    fakecatalog (Table): Table of the fake sources that were injected
    """
    test_image = test_hdu[1].data  # Assumes JWST data model
    psfshape = test_psf[0].data.shape
    oversample = test_psf[0].header["OVERSAMP"]
    w = wcs.WCS(test_hdu[1].header)
    pixel_area = w.proj_plane_pixel_area()
    fluxconv = (1 * u.uJy / pixel_area).to(u.MJy / u.sr)
    xfake = np.random.rand(n_sources) * (test_image.shape[1] - 1)
    yfake = np.random.rand(n_sources) * (test_image.shape[0] - 1)
    # Keep where data are valid
    keep = test_image[yfake.astype(int), xfake.astype(int)] != 0
    keep = keep & (test_image[yfake.astype(int), xfake.astype(int)] != np.nan)
    # Don't get so close to edge that PSF will run over image edge
    keep = keep & (
        (xfake - psfshape[1] // 2 > 0)
        & (xfake + psfshape[1] // 2 - 1 < test_image.shape[1])
        & (yfake - psfshape[0] // 2 > 0)
        & (yfake + psfshape[0] // 2 - 1 < test_image.shape[0])
    )
    xfake = xfake[keep]
    yfake = yfake[keep]
    charflux = np.log(charflux.to(u.uJy).value)  # characteristic flux
    fluxfake = np.exp(np.random.randn(len(xfake)) * fluxspread + charflux) * u.uJy
    fakecatalog = Table()
    fakecatalog["xcentroid"] = xfake
    fakecatalog["ycentroid"] = yfake
    ra, dec = w.all_pix2world(xfake, yfake, 0)
    fakecatalog["RA"] = ra
    fakecatalog["DEC"] = dec
    fakecatalog["flux"] = fluxfake
    fakeimage = np.copy(test_image)
    for ii in np.arange(len(fluxfake)):
        xx = xfake[ii]
        yy = yfake[ii]
        xfrac = xx - int(xx)
        yfrac = yy - int(yy)
        offsetx = np.arange(psfshape[1])
        offsety = np.arange(psfshape[0])
        xarr, yarr = np.meshgrid(offsety, offsetx)
        psf_shift = map_coordinates(
            test_psf[0].data,
            [
                yarr - (yfrac) * oversample - oversample / 2,
                xarr - (xfrac) * oversample - oversample / 2,
            ],
            order=1,
            mode="constant",
            cval=0.0,
        )
        # group into blocks of oversamp x oversamp
        psf_shift = np.add.reduceat(
            psf_shift, np.arange(0, psf_shift.shape[0], oversample), axis=0
        )
        psf_shift = np.add.reduceat(
            psf_shift, np.arange(0, psf_shift.shape[1], oversample), axis=1
        )
        psf_shift *= fluxconv.value * fluxfake[ii].value
        fakeimage[
            int(yy) - psf_shift.shape[0] // 2 : int(yy)
            + psf_shift.shape[0] // 2
            + psf_shift.shape[0] % 2,
            int(xx) - psf_shift.shape[1] // 2 : int(xx)
            + psf_shift.shape[1] // 2
            + psf_shift.shape[1] % 2,
        ] += psf_shift
    if return_hdu:
        hdulist = fits.HDUList(
            [
                fits.PrimaryHDU(fakeimage, header=test_hdu[1].header),
                fits.BinTableHDU(fakecatalog),
            ]
        )
        return (hdulist, fakecatalog)
    else:
        return fakeimage, fakecatalog


def constrained_diffusion(inputdata, err_rel=3e-2, n_scales=None):
    """
    Apply a constrained diffusion process to the input data.

    Parameters:
    inputdata (ndarray): Input data to be processed.
    err_rel (float, optional): Relative error for the diffusion process.
                               Defaults to 3e-2.
    n_scales (int, optional): Number of scales to use in the diffusion process.
                              If None, it will be computed based on the input data
                              shape. Defaults to None.

    Returns:
    tuple: A tuple containing the scalecube (a cube of images representing the
           diffusion process at different scales) and the final diffused data.
    """
    data = np.copy(inputdata)
    ntot = min(int(np.log(min(data.shape)) / np.log(2) - 1), n_scales or float("inf"))
    scalecube = np.zeros((ntot, *data.shape))

    for i in range(ntot):
        channel_image = np.zeros_like(data)
        scale_end, scale_begin = 2 ** (i + 1), 2**i
        t_end, t_begin = scale_end**2 / 2, scale_begin**2 / 2
        delta_t_max = t_begin * (0.1 if i == 0 else err_rel)
        niter = int((t_end - t_begin) / delta_t_max + 0.5)
        delta_t = (t_end - t_begin) / niter
        kernel_size = np.sqrt(2 * delta_t)

        for _ in range(niter):
            if kernel_size > 5:
                smooth_image = convolve_fft(data, Gaussian2DKernel(kernel_size))
            else:
                smooth_image = gf(data, kernel_size, mode="constant", cval=0.0)

            sm_image_min, sm_image_max = (
                np.minimum(data, smooth_image),
                np.maximum(data, smooth_image),
            )
            diff_image = np.zeros_like(data)
            pos_1, pos_2 = (
                np.where((data - sm_image_min > 0) & (data > 0)),
                np.where((data - sm_image_max < 0) & (data < 0)),
            )
            diff_image[pos_1], diff_image[pos_2] = (
                data[pos_1] - sm_image_min[pos_1],
                data[pos_2] - sm_image_max[pos_2],
            )
            channel_image += diff_image
            data -= diff_image

        scalecube[i] = channel_image
    return scalecube, data


def phot_catalog(
    image, wcs, psf_fwhm, xcentroids=None, ycentroids=None, filter_diffuse=False
):
    """
    Generate a photometric catalog from an image.

    Parameters:
    image (ndarray): Input image.
    wcs (WCS): World Coordinate System of the image.
    psf_fwhm (float): Full width at half maximum of the point spread function.
    xcentroids (ndarray, optional): X coordinates of the sources.
                If None, they will be computed. Defaults to None.
    ycentroids (ndarray, optional): Y coordinates of the sources.
               If None, they will be computed. Defaults to None.
    filter_diffuse (bool, optional): If True, apply a diffusion
                filter to the image. Defaults to False.

    Returns:
    Table: Photometric catalog with columns for xcentroid, ycentroid,
           aperflux (aperture flux), bkgflux (background flux), RA, DEC, and skycoord.
    """
    if xcentroids is not None and ycentroids is not None:
        sources = Table()
        sources["xcentroid"] = xcentroids
        sources["ycentroid"] = ycentroids
        compact = image
        std = mad_std(compact[compact != 0])
    else:
        if filter_diffuse:
            filtered_data = constrained_diffusion(image, n_scales=3)
            pivot_scale = 1
            compact = np.sum(filtered_data[0][0:pivot_scale], axis=0)
        # # diffuse = np.sum(filtered_data[0][pivot_scale:], axis=0) + filtered_data[1]
        else:
            compact = image
        std = mad_std(compact[compact != 0])
        iraffind = IRAFStarFinder(fwhm=2.0, threshold=5.0 * std, minsep_fwhm=0.0)
        sources = iraffind(compact)
    aper = CircularAperture(
        np.c_[sources["xcentroid"], sources["ycentroid"]], r=psf_fwhm * 2.5
    )
    bkaper = CircularAnnulus(
        np.c_[sources["xcentroid"], sources["ycentroid"]],
        r_in=psf_fwhm * 4,
        r_out=psf_fwhm * 5,
    )
    sourcephot = aperture_photometry(image, aper)
    bkgphot = ApertureStats(image, bkaper)
    bkg_flux = bkgphot.median * aper.area
    sources["aperflux"] = sourcephot["aperture_sum"]
    sources["bkgflux"] = bkg_flux
    ra, dec = wcs.all_pix2world(sources["xcentroid"], sources["ycentroid"], 0)
    sources["RA"] = ra
    sources["DEC"] = dec
    sources["skycoord"] = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    return sources


def completeness_analysis(
    fakefiles, catfiles, compfrac=0.99, verbose=True, match_radius=0.1 * u.arcsec
):
    """
    Given a list of sources with fakes injected and a list of catalogs for those sources,
    measure the completeness limit.
    """
    sourcecats = []
    for thisfile, catfile in zip(fakefiles, catfiles):
        fake_catalog = Table.read(catfile)
        test_image = fits.open(thisfile)
        true_fakes = Table(test_image[1].data)
        if "skycoord" not in fake_catalog.keys():
            fake_catalog["skycoord"] = SkyCoord(
                ra=fake_catalog["RA"], dec=fake_catalog["DEC"], unit=(u.deg, u.deg)
            )
        if "skycoord" not in true_fakes.keys():
            true_fakes["skycoord"] = SkyCoord(
                ra=true_fakes["RA"], dec=true_fakes["DEC"], unit=(u.deg, u.deg)
            )
        _, sep2d, _ = match_coordinates_sky(
            true_fakes["skycoord"], fake_catalog["skycoord"]
        )
        matched = sep2d < match_radius
        true_fakes["DETECT"] = np.zeros(len(true_fakes), dtype=bool)
        true_fakes["DETECT"][matched] = True
        sourcecats.append(true_fakes)
    sourcecats = vstack(sourcecats)
    df = sourcecats.to_pandas()
    df["LOGFLUX"] = np.log10(df["flux"])
    df["DETECTION"] = df["DETECT"].astype(int)
    compmodel = logit("DETECTION ~ LOGFLUX", df).fit()
    complim = 1e1 ** (
        -(np.log(1 / compfrac - 1) + compmodel.params["Intercept"])
        / compmodel.params["LOGFLUX"]
    )
    if verbose:
        print(compmodel.summary())
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.semilogx(sourcecats["flux"], sourcecats["DETECT"], "o", markersize=3)
        x = np.geomspace(1e-1, 1e4, 300)
        ax.plot(
            x, 1 / (1 + np.exp(-compmodel.params["Intercept"]
                               -compmodel.params["LOGFLUX"] * np.log10(x))),
            color="r",
        )
        ax.set_xscale("log")
        shortname = os.path.basename(fakefiles[0]).replace(".fits", "")
        fig.savefig(f"completeness_{shortname}.png")
        print(f"{compfrac*100}% Completeness limit: {complim:5.2f} µJy")
    return complim, compmodel


def completeness_limit(
    FILENAME,
    oversample=4,
    match_radius=0.1 * u.arcsec,
    n_batch=5,
    n_sources=1000,
    write_fake_files=False,
    write_catalogs=False,
    compfrac=0.99,
    verbose=True,
    doplot=False,
):
    """
    Calculate the completeness limit of an image.

    Parameters:
    FILENAME (str): Path to the FITS file to be processed.
    oversample (int, optional): Oversampling factor for the PSF calculation. Defaults to 4.
    match_radius (Quantity, optional): Maximum distance for a source to be considered a match.
                 Defaults to 0.1 arcsec.
    n_batch (int, optional): Number of batches to process. Defaults to 5.
    n_sources (int, optional): Number of sources to generate in the fake image.
              Defaults to 1000.
    write_fake_files (bool, optional): If True, write images with fake sources.
              Defaults to False.
    write_catalogs (bool, optional): If True, write catalogs of the fake sources.
              Defaults to False.
    compfrac (float, optional): Completeness fraction to calculate the limit for.
              Defaults to 0.99.
    verbose (bool, optional): If True, print additional information and plot the
            completeness curve. Defaults to True.

    Returns:
    tuple: A tuple containing the completeness limit and the fitted logistic regression model.
    """
    test_image = fits.open(FILENAME)
    w = wcs.WCS(test_image[1].header)
    camera = test_image[0].header["INSTRUME"].strip()
    if camera == "MIRI":
        miri = webbpsf.MIRI()
        miri.filter = test_image[0].header["FILTER"]
        test_psf = miri.calc_psf(oversample=oversample)
    elif camera == "NIRCAM":
        nircam = webbpsf.NIRCam()
        nircam.filter = test_image[0].header["FILTER"]
        test_psf = nircam.calc_psf(oversample=oversample)

    pixel_area = w.proj_plane_pixel_area()
    fluxconv = (1 * u.uJy / pixel_area).to(u.MJy / u.sr)

    # This is a little low because of the PSF shape
    psf_fwhm = 2 * np.sqrt(
        np.sum(test_psf[0].data >= (0.5 * test_psf[0].data.max())) / np.pi
    )
    sourcecats = []
    for batches in range(n_batch):
        fakeimage, true_fakes = fake_image(
            test_image, test_psf, n_sources, return_hdu=write_fake_files
        )
        if write_fake_files:
            outfile = os.path.basename(
                FILENAME.replace(".fits", f"_fake_{batches}.fits")
            )
            fakeimage.writeto(outfile, overwrite=True)
            fakeimage = fakeimage[0].data
        fake_catalog = phot_catalog(fakeimage, w, psf_fwhm)
        if write_catalogs:
            fake_catalog.write(
                os.path.basename(FILENAME.replace(".fits", f"_fakecat_{batches}.fits")),
                format="fits",
            )
        truefakes_catalog = phot_catalog(
            fakeimage,
            w,
            psf_fwhm,
            xcentroids=true_fakes["xcentroid"],
            ycentroids=true_fakes["ycentroid"],
        )
        match_idx, sep2d, _ = match_coordinates_sky(
            truefakes_catalog["skycoord"], fake_catalog["skycoord"]
        )
        matched = sep2d < match_radius
        truefakes_catalog["DETECT"] = np.zeros(len(truefakes_catalog), dtype=bool)
        truefakes_catalog["DETECT"][matched] = True
        truefakes_catalog["MEASURED_FLUX"] = np.zeros(len(truefakes_catalog)) * np.nan
        truefakes_catalog["MEASURED_FLUX"][matched] = (
            fake_catalog["aperflux"][match_idx[matched]]
            - fake_catalog["bkgflux"][match_idx[matched]]
        ) / fluxconv
        truefakes_catalog["FLUX_AT_LOCATION"] = (
            truefakes_catalog["aperflux"] - truefakes_catalog["bkgflux"]
        ) / fluxconv
        truefakes_catalog["TRUE_FLUX"] = true_fakes["flux"]
        sourcecats.append(truefakes_catalog)
    sourcecats = vstack(sourcecats)
    df = sourcecats.to_pandas()
    df["LOGFLUX"] = np.log10(df["TRUE_FLUX"])
    df["LOGBG"] = np.log10(df["bkgflux"])
    df["DETECTION"] = df["DETECT"].astype(int)
    compmodel = logit("DETECTION ~ LOGFLUX + bkgflux", df).fit()
    complim = 1e1 ** (
        -(np.log(1 / compfrac - 1) + compmodel.params["Intercept"])
        / compmodel.params["LOGFLUX"]
    )
    if verbose:
        print(compmodel.summary())
        print(f"{compfrac*100}% Completeness limit: {complim:5.2f} µJy")
    if doplot:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.semilogx(sourcecats["TRUE_FLUX"], sourcecats["DETECT"], "o", markersize=3)
        x = np.geomspace(1e-1, 1e4, 300)
        ax.plot(x, 1 / (1 + np.exp(-compmodel.params["Intercept"]
                                   - compmodel.params["LOGFLUX"] * np.log10(x))),
            color="r",
        )
        ax.set_xscale("log")
        shortname = os.path.basename(FILENAME).replace(".fits", "")
        fig.savefig(f"completeness_{shortname}.png")
    return complim, compmodel


def main():
    try:
        FILENAME = sys.argv[1]
    except IndexError:
        print("Usage: jwst_complim <Level3_fits_filename> <completeness_fraction_0_to_1>")
        sys.exit(1)
    if len(sys.argv) > 2:
        compfrac = float(sys.argv[2])
    else:
        compfrac = 0.99    
    complim, _ = completeness_limit(
        FILENAME,
        n_sources=1000,
        write_fake_files=False,
        write_catalogs=False,
        compfrac=compfrac
    )
    print(f"For file {os.path.basename(FILENAME)}, the {compfrac*100}% completeness limit is:")
    print(f"{complim:5.2f} µJy")
