from typing import List
import hyperspy.api as hs
import numpy as np
from sigma.utils.load import SEMDataset
from hyperspy.signals import Signal2D, Signal1D
from hyperspy._signals.signal2d import Signal2D, Signal2D
from hyperspy._signals.eds_tem import EDSTEMSpectrum
from .base import BaseDataset

# def radial_integral(img2d, r):
#     height, width = img2d.shape
#     # Compute the center of the image
#     center_x, center_y = width // 2, height // 2
#     # Create a meshgrid of coordinates
#     y, x = np.ogrid[:height, :width]
#     # Calculate the radial distance of each point from the center
#     distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
#     # Find points that are close to the given radius `r`
#     mask = np.abs(distance_from_center - r) < 0.5  # A tolerance of 0.5 to account for pixels around the radius
#     return img2d[mask].sum()

# def list_radial_integral(img2d, r_list):
#     return [radial_integral(img2d, r) for r in r_list]

class SEDDataset(BaseDataset):
    def __init__(self, file_path: str, n_root = 1, integral_type = "radial", unit = "k_A^-1", beam_energy = 200, npt = 100):
        super().__init__(file_path)

        self.base_dataset.data = self.base_dataset.data**(1 / n_root)
        # dp_height, dp_width = self.base_dataset.data.shape[2]
        self.nav_img = Signal2D(self.base_dataset.data.sum(axis = (-1, -2)))
        # do radial integral here
        print("computing radial average")
        if integral_type == "radial":
            self.spectra = self.base_dataset.radial_average()
        elif integral_type == "azimuthal":
            self.base_dataset.unit = unit
            self.base_dataset.beam_energy = beam_energy
            self.base_dataset.set_ai()
            self.spectra = self.base_dataset.get_azimuthal_integral1d(inplace = False, npt = npt)
        else:
            ValueError("integral type not found")

        self.spectra.change_dtype("float32")
        self.spectra_raw = self.spectra.deepcopy()
        # feature list and feature dict seems not used???
        self.feature_list = ["Default"]
        self.feature_dict = {el: i for (i, el) in enumerate(self.feature_list)}


    def set_axes_scale(self, scale:float):
        """
        Set the scale for the energy axis. 

        Parameters
        ----------
        scale : float
            The scale of the energy axis. For example, given a data set with 1500 data points corresponding to 0-15 keV, the scale should be set to 0.01.

        """
        self.spectra.axes_manager["Energy"].scale = scale
    
    def set_axes_offset(self, offset:float):
        """
        Set the offset for the energy axis. 

        Parameters
        ----------
        offset : float
            the offset of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].offset = offset

    def set_axes_unit(self, unit:str):
        """
        Set the unit for the energy axis. 

        Parameters
        ----------
        unit : float
            the unit of the energy axis. 

        """
        self.spectra.axes_manager["Energy"].unit = unit
    
    def remove_NaN(self):
        """
        Remove the pixels where no values are stored.
        """
        index_NaN = np.argwhere(np.isnan(self.spectra.data[:,0,0]))[0][0]
        self.nav_img.data = self.nav_img.data[:index_NaN-1,:]
        self.spectra.data = self.spectra.data[:index_NaN-1,:,:]

        if self.nav_img_bin is not None:
            self.nav_img_bin.data = self.nav_img_bin.data[:index_NaN-1,:]
        if self.spectra_bin is not None:
            self.spectra_bin.data = self.spectra_bin.data[:index_NaN-1,:,:]

    def normalisation(self, norm_list=[]):
        self.normalised_elemental_data = self.get_feature_maps(self.feature_list)