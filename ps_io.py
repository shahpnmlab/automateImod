from typing import Union
from pathlib import Path
import mrcfile

from pydantic import BaseModel


class TiltSeries(BaseModel):
    path_to_data: Union[str, Path] = None
    name: str = None
    rawtlt: str = None
    mdoc: str = None
    tilt_axis_ang: Union[str, float] = None
    binval: Union[str, int] = None
    z_height: Union[str, int] = None
    extension: str = None
    patch_size: Union[str, int] = None

    def tilt_dir_name(self):
        if self.path_to_data and self.name:
            return Path(self.path_to_data) / self.name
        return None


class Tomogram(BaseModel):
    path_to_data: Union[str, Path] = None
    name: str = None
    extension: str = None
    thickness: str = None
    binval: str = None

    def tilt_dir_name(self):
        if self.path_to_data and self.name:
            return Path(self.path_to_data) / self.name
        return None

def read_mrc(mrcin):
    with mrcfile.open(mrcin) as mrc:
        data = mrc.data
        dimX = str(mrc.header.nx)
        dimY = str(mrc.header.ny)
        pixel_nm = (mrc.voxel_size.x) / 10
    return data, pixel_nm, dimX, dimY
