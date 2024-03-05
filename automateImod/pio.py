import mrcfile
import mdocfile
import numpy as np
from pathlib import Path, PureWindowsPath
from pydantic import BaseModel
from typing import Union, List


class TiltSeries(BaseModel):
    path_to_ts_data: Union[str, Path] = None
    path_to_xml_data: Union[str, Path] = None
    path_to_mdoc_data: Union[str, Path] = None
    basename: str = None
    tilt_name: str = None
    rawtlt: str = None
    mdoc: str = None
    xml: str = None
    tilt_axis_ang: Union[str, float] = None
    binval: Union[str, int] = None
    z_height: Union[str, int] = None
    extension: str = None
    patch_size: Union[str, int] = None
    tilt_frames: List[str] = []
    tilt_angles: List[float] = []
    tilt_dir_name: Path = None

    def __init__(self, **data):
        super().__init__(**data)
        # self.get_extension()
        if self.path_to_ts_data and self.basename:
            # self.tilt_dir_name = Path(f"{data['path_to_ts_data']}/{data['basename']}")
            self.tilt_dir_name = Path(self.path_to_ts_data) / self.basename
            self.get_extension()
            self.tilt_name = f"{self.basename}.{self.extension}" if self.extension else None
            self.rawtlt = f'{self.basename}.rawtlt'
            self.mdoc = f'{self.basename}.mdoc'
            self.xml = f'{self.basename}.xml'
        self.tilt_angles = self.read_rawtlt_file()
        self.read_mdoc_file()
    def get_extension(self):
        if self.tilt_dir_name:  # Ensure tilt_dir_name is not None
            st_path = Path(f"{self.tilt_dir_name}/{self.basename}.st")
            mrc_path = Path(f"{self.tilt_dir_name}/{self.basename}.mrc")
            if mrc_path.is_file():
                self.extension = "mrc"
            elif st_path.is_file():  # Use elif to prioritize .mrc over .st if both exist
                self.extension = "st"

    def get_mrc_path(self):
        return self.tilt_dir_name / self.tilt_name if self.tilt_dir_name and self.tilt_name else None

    def get_mdoc_path(self):
        return Path(self.path_to_mdoc_data) / self.mdoc if self.path_to_mdoc_data and self.mdoc else None

    def get_xml_path(self):
        return Path(self.path_to_xml_data) / self.xml if self.path_to_xml_data and self.xml else None

    def get_rawtlt_path(self):
        return Path(self.tilt_dir_name) / self.rawtlt if self.path_to_ts_data and self.rawtlt else None

    def read_mdoc_file(self):
        mdoc_path = self.get_mdoc_path()
        if mdoc_path and mdoc_path.exists():
            md = mdocfile.read(mdoc_path)
            if len(self.tilt_angles):
                # Determine the sorting order (ascending or descending)
                ascending = self.tilt_angles[0] < self.tilt_angles[-1]
                # Sort the mdoc data to match the tilt_angles order
                sorted_md = md.sort_values("TiltAngle", ascending=ascending)
                sorted_md["TiltAngle"] = sorted_md["TiltAngle"].round(1)
                filtered_df = sorted_md[sorted_md["TiltAngle"].isin(self.tilt_angles)]
                self.tilt_frames = filtered_df["SubFramePath"].apply(lambda x: PureWindowsPath(x).name).to_list()
        else:
            print(f"Could not find {self.mdoc} in {self.path_to_mdoc_data}.")

    def read_rawtlt_file(self):
        tilt_angles = []
        if self.rawtlt:
            rawtlt_path = self.get_rawtlt_path()
            if rawtlt_path.exists():
                tilt_angles = np.loadtxt(rawtlt_path)
            else:
                print(f"Could not find {self.rawtlt} in {rawtlt_path}.")
        return tilt_angles

    def remove_frames(self, indices: List[int]):
        """
        Remove frames by indices from tilt_frames and tilt_angles using NumPy.
        """
        if isinstance(self.tilt_angles, np.ndarray):
            # Use numpy.delete to remove elements from numpy arrays
            self.tilt_angles = np.delete(self.tilt_angles, indices)
            # For tilt_frames, since it's a list, first ensure it's a numpy array, perform deletion, then convert back if necessary
            self.tilt_frames = np.delete(np.array(self.tilt_frames), indices).tolist()
        else:
            # If tilt_angles is somehow not a numpy array, revert to list-based removal
            sorted_indices = sorted(indices, reverse=True)
            for index in sorted_indices:
                if index < len(self.tilt_frames):
                    del self.tilt_frames[index]
                if index < len(self.tilt_angles):
                    del self.tilt_angles[index]

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


if __name__ == '__main__':
    a = TiltSeries(path_to_ts_data="/Users/ps/data/wip/automateImod/example_data/Frames/imod/",
                   path_to_xml_data="None",
                   path_to_mdoc_data="/Users/ps/data/wip/automateImod/example_data/mdoc",
                   basename="map-26-A4_ts_002",
                   binval="5",
                   tilt_axis_ang="84.7",
                   patch_size="350")
    print(a.read_rawtlt_file())
