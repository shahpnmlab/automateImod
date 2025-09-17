import mrcfile
import starfile
import mdocfile
import xml.etree.ElementTree as ET
import logging

from pydantic import BaseModel
from typing import Union, List, Optional
from pathlib import Path
import numpy as np


class TiltSeries(BaseModel):
    path_to_ts_data: Union[str, Path]
    path_to_xml_data: Optional[Union[str, Path]] = None
    path_to_mdoc_data: Optional[Union[str, Path]] = None
    path_to_tomostar: Optional[Union[str, Path]] = None
    basename: str
    tilt_name: Optional[str] = None
    rawtlt: Optional[str] = None
    mdoc: Optional[str] = None
    xml: Optional[str] = None
    tomostar: Optional[str] = None
    tilt_axis_ang: Union[str, float]
    pixel_size: Union[str, float] = None
    binval: Union[str, int]
    z_height: Optional[Union[str, int]] = None
    extension: Optional[str] = None
    patch_size: Union[str, int] = None
    patch_size_ang: Union[str, float]
    tilt_frames: List[str] = []
    tilt_angles: Union[List[float], np.ndarray] = []
    tilt_dir_name: Optional[Path] = None
    axis_angles: List[float] = []
    doses: List[float] = []
    use_tilt: List[bool] = []
    axis_offset_x: List[float] = []
    axis_offset_y: List[float] = []
    plane_normal: Optional[str] = None
    magnification_correction: Optional[str] = None
    unselect_filter: Optional[bool] = None
    unselect_manual: Optional[str] = None
    ctf_resolution_estimate: Optional[float] = None
    removed_indices: Optional[List[int]] = []
    # Make dimX and dimY optional with None as default
    dimX: Optional[Union[str, int]] = None
    dimY: Optional[Union[str, int]] = None
    logger: Optional[logging.Logger] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # If a logger is not provided, create a default one
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                self.logger.addHandler(logging.StreamHandler())
                self.logger.setLevel(logging.INFO)

        if self.path_to_ts_data and self.basename:
            self.tilt_dir_name = Path(self.path_to_ts_data) / self.basename
            self.rawtlt = f"{self.basename}.rawtlt"
            self.mdoc = f"{self.basename}.mdoc"
            self.xml = f"{self.basename}.xml"
            self.get_extension()
            self.tomostar = f"{self.basename}.tomostar"
            self.tilt_frames = []
            self.tilt_angles = []

            # Initialize basic properties
            self.read_rawtlt_file()
            self.tilt_name = f"{self.basename}.{self.extension}"

            # Get dimensions and pixel size from MRC file
            self._get_ts_dim()  # This will set dimX, dimY, and pixel_size

            # Continue with other initialization steps
            self._convert_patch_size()
            self.read_mdoc_file()

            if self.path_to_tomostar:
                self.read_tomostar_file()

    def _get_ts_dim(self):
        """Get dimensions and pixel size from the MRC file."""
        try:
            mrc_path = self.get_mrc_path()
            if mrc_path and mrc_path.exists():
                with mrcfile.open(mrc_path) as mrc:
                    self.dimX = mrc.data.shape[-2]
                    self.dimY = mrc.data.shape[-1]
                    self.pixel_size = float(mrc.voxel_size.x)
                    self.logger.info(f"Dimensions: {self.dimX} x {self.dimY}")
                    self.logger.info(f"Pixel size: {self.pixel_size:.2f} Å")
            else:
                raise FileNotFoundError(f"MRC file not found: {mrc_path}")
        except Exception as e:
            self.logger.error(f"Error reading MRC file: {e}")
            # Set default values if reading fails
            self.dimX = None
            self.dimY = None
            self.pixel_size = None

    def _get_pixel_size(self):
        """Get pixel size in angstroms from the MRC file."""
        try:
            mrc_path = self.get_mrc_path()
            if mrc_path and mrc_path.exists():
                with mrcfile.open(mrc_path) as mrc:
                    # Convert from nm to angstroms (1 nm = 10 Å)
                    self.pixel_size = mrc.voxel_size.x
                    self.logger.info(f"Pixel size: {self.pixel_size:.2f} Å")
            else:
                raise FileNotFoundError(f"MRC file not found: {mrc_path}")
        except Exception as e:
            self.logger.error(f"Error reading pixel size: {e}")
            self.pixel_size = None

    def _convert_patch_size(self):
        """Convert patch size from angstroms to pixels."""
        if self.pixel_size and self.patch_size_ang:
            # Convert patch_size_ang to float if it's a string
            patch_ang = float(self.patch_size_ang)
            # Calculate binned pixel size in angstroms
            binned_px = float(self.pixel_size) * int(self.binval)
            # Calculate pixels and round to nearest integer
            self.patch_size = int(round(patch_ang / binned_px))
            self.logger.info(
                f"Patch size converted from {patch_ang} Å to {self.patch_size} pixels (binned pixel size: {binned_px:.2f} Å)"
            )
        else:
            self.logger.warning(
                "Could not convert patch size to pixels. Missing pixel size or patch size in angstroms."
            )

    def read_tomostar_file(self):
        tomostar_path = (
            Path(self.path_to_tomostar) / self.tomostar
            if self.path_to_tomostar and self.tomostar
            else None
        )
        if tomostar_path and tomostar_path.exists():
            try:
                tomostar_data = starfile.read(tomostar_path)

                # Convert wrpMovieName entries to Path objects and extract filenames
                self.tilt_frames = [
                    Path(movie_name).name
                    for movie_name in tomostar_data["wrpMovieName"]
                ]
                self.tilt_angles = np.array(tomostar_data["wrpAngleTilt"])
                self.axis_angles = tomostar_data["wrpAxisAngle"]
                self.doses = tomostar_data["wrpDose"]

                self.logger.info(
                    f"Processed {len(self.tilt_frames)} tilt frames from tomostar file."
                )
            except KeyError as e:
                self.logger.error(f"Error reading tomostar file: Missing key {e}")
            except Exception as e:
                self.logger.error(f"Error reading tomostar file: {e}")
        else:
            self.logger.warning(f"Could not find {self.tomostar} in {self.path_to_tomostar}.")

    def read_rawtlt_file(self):
        rawtlt_path = self.get_rawtlt_path()
        if rawtlt_path and rawtlt_path.exists():
            self.tilt_angles = np.loadtxt(rawtlt_path)
            self.logger.info(f"Loaded {len(self.tilt_angles)} tilt angles from rawtlt file.")
        else:
            self.logger.warning(f"Could not find {self.rawtlt} in {self.tilt_dir_name}.")

    def read_mdoc_file(self):
        mdoc_path = self.get_mdoc_path()
        if mdoc_path and mdoc_path.exists():
            md = mdocfile.read(mdoc_path)
            if not self.tilt_frames:  # Only update if tilt_frames is empty
                self.tilt_frames = (
                    md["SubFramePath"].apply(lambda x: Path(x).name).tolist()
                )
            self.logger.info(f"Processed {len(self.tilt_frames)} tilt frames from mdoc file.")
        else:
            self.logger.warning(f"Could not find {self.mdoc} in {self.path_to_mdoc_data}.")

    def get_extension(self):
        if self.tilt_dir_name:  # Ensure tilt_dir_name is not None
            st_path = Path(f"{self.tilt_dir_name}/{self.basename}.st")
            mrc_path = Path(f"{self.tilt_dir_name}/{self.basename}.mrc")
            if mrc_path.is_file():
                self.extension = "mrc"
            elif (
                st_path.is_file()
            ):  # Use elif to prioritize .mrc over .st if both exist
                self.extension = "st"

    def get_mrc_path(self):
        return self.tilt_dir_name / self.tilt_name

    def get_mdoc_path(self):
        """
        Get the path to the mdoc file, checking both .mdoc and .mrc.mdoc patterns.
        Returns the first existing file, or None if neither exists.
        """
        if not (self.path_to_mdoc_data and self.basename):
            return None

        mrc_mdoc = Path(self.path_to_mdoc_data) / f"{self.basename}.mrc.mdoc"
        if mrc_mdoc.exists():
            return mrc_mdoc
        mdoc = Path(self.path_to_mdoc_data) / f"{self.basename}.mdoc"
        if mdoc.exists():
            return mdoc
        return mrc_mdoc

    def get_xml_path(self):
        return (
            Path(self.path_to_xml_data) / self.xml
            if self.path_to_xml_data and self.xml
            else None
        )

    def get_rawtlt_path(self):
        return (
            Path(self.tilt_dir_name) / self.rawtlt
            if self.path_to_ts_data and self.rawtlt
            else None
        )

    def remove_frames(self, indices: List[int]):
        """
        Remove frames by indices from tilt_frames and tilt_angles using NumPy.
        """
        if isinstance(self.tilt_angles, np.ndarray):
            # Use numpy.delete to remove elements from numpy arrays
            # For tilt_frames, since it's a list, first ensure it's a numpy array, perform deletion, then convert back if necessary
            self.tilt_angles = np.delete(self.tilt_angles, indices)
            self.tilt_frames = np.delete(np.array(self.tilt_frames), indices).tolist()
        else:
            # If tilt_angles is somehow not a numpy array, revert to list-based removal
            sorted_indices = sorted(indices, reverse=True)
            for index in sorted_indices:
                if index < len(self.tilt_frames):
                    del self.tilt_frames[index]
                if index < len(self.tilt_angles):
                    del self.tilt_angles[index]

    def read_xml_file(self):
        xml_path = (
            Path(self.path_to_xml_data) / self.xml
            if self.path_to_xml_data and self.xml
            else None
        )
        if xml_path and xml_path.exists():
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Parse XML attributes
                self.plane_normal = root.get("PlaneNormal")
                self.magnification_correction = root.get("MagnificationCorrection")
                self.unselect_filter = root.get("UnselectFilter") == "True"
                self.unselect_manual = root.get("UnselectManual")
                self.ctf_resolution_estimate = float(
                    root.get("CTFResolutionEstimate", 0)
                )

                # Parse XML elements
                self.use_tilt = [
                    x.lower() == "true" for x in root.find("UseTilt").text.split()
                ]
                self.axis_offset_x = [
                    float(x) for x in root.find("AxisOffsetX").text.split()
                ]
                self.axis_offset_y = [
                    float(x) for x in root.find("AxisOffsetY").text.split()
                ]

                self.logger.info("Successfully parsed XML file.")
            except Exception as e:
                self.logger.error(f"Error reading XML file: {e}")
        else:
            self.logger.warning(f"Could not find {self.xml} in {self.path_to_xml_data}.")


class Tomogram(BaseModel):
    path_to_data: Union[str, Path] = None
    basename: str = None
    extension: str = None
    thickness: str = None
    binval: str = None
    tilt_dir_name: Path = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.path_to_data and self.basename:
            self.tilt_dir_name = Path(self.path_to_data) / self.basename


def read_mrc(mrcin):
    with mrcfile.open(mrcin) as mrc:
        data = mrc.data
        dimX = str(mrc.header.nx)
        dimY = str(mrc.header.ny)
        pixel_nm = mrc.voxel_size.x / 10
    return data, pixel_nm, dimX, dimY


if __name__ == "__main__":
    a = TiltSeries(
        path_to_ts_data="/Users/ps/data/wip/automateImod/example_data/Frames/imod/",
        path_to_xml_data="None",
        path_to_mdoc_data="/Users/ps/data/wip/automateImod/example_data/mdoc",
        basename="map-26-A4_ts_002",
        binval="5",
        tilt_axis_ang="84.7",
        patch_size_ang="350",  # Changed patch_size to patch_size_ang to match class definition
    )
    print(a.read_rawtlt_file())


"""
align-tilts --ts-data-path tiltstack  --ts-mdoc-path tiltstack/mdocs --ts-basename lamella9_ts_002 --ts-bin 6 --ts-patch-size 300 --ts-tilt-axis 84.7
"""
