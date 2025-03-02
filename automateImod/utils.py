import shutil
import subprocess
import mrcfile
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional

import automateImod.calc as calc
import automateImod.pio as io

# Configure logger
logger = logging.getLogger(__name__)


def setup_logger(log_level: str = "INFO"):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get log level from string
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
    )

    # Configure automateImod logger
    logger = logging.getLogger("automateImod")
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(console_handler)


def detect_large_shifts_afterxcorr(
    coarse_align_prexg: str,
    pixel_size_nm: float,
    image_size: Tuple[int, int],
    min_fov_fraction: float = 0.7,
) -> List[int]:
    """
    Detect frames that have shifted beyond the acceptable field of view (FOV).

    Args:
        coarse_align_prexg: Path to the prexg file containing shift information
        pixel_size_nm: Pixel size in nanometers
        image_size: Image dimensions (width, height) in pixels
        min_fov_fraction: Minimum required overlap as a fraction of FOV (default: 0.7)

    Returns:
        List of indices of frames with unacceptable shifts
    """
    # Convert nm to Angstrom
    pixel_size_ang = pixel_size_nm * 10

    # Read shift data from prexg file
    prexg_data = []
    try:
        with open(coarse_align_prexg, "r") as file:
            for line in file:
                numbers = [float(num) for num in line.split()]
                prexg_data.append(numbers[-2:])  # Last two numbers are X,Y shifts
    except (IOError, ValueError) as e:
        logger.error(f"Error reading prexg file: {e}")
        return []

    prexg_data = np.array(prexg_data)

    # Convert shifts to Angstroms
    shifts_ang = prexg_data * pixel_size_ang

    # Calculate FOV dimensions in Angstroms
    fov_width_ang = image_size[0] * pixel_size_ang
    fov_height_ang = image_size[1] * pixel_size_ang

    # Calculate maximum allowed shift (as percentage of FOV)
    max_shift_x = (1 - min_fov_fraction) * fov_width_ang
    max_shift_y = (1 - min_fov_fraction) * fov_height_ang

    # Find frames where shifts exceed the maximum allowed in either direction
    large_shift_indices = []
    for idx, (shift_x, shift_y) in enumerate(shifts_ang):
        if abs(shift_x) > max_shift_x or abs(shift_y) > max_shift_y:
            large_shift_indices.append(idx)

    return large_shift_indices


def remove_bad_tilts(
    ts: io.TiltSeries, im_data: np.ndarray, pixel_nm: float, bad_idx: List[int]
) -> None:
    """
    Remove problematic tilts from the tilt series data and rawtlt file.

    Args:
        ts: TiltSeries object
        im_data: Image data array
        pixel_nm: Pixel size in nanometers
        bad_idx: List of indices to remove
    """
    if not bad_idx:
        logger.info("No bad tilts to remove")
        return

    # Convert nm to angstroms for MRC file
    angpix = pixel_nm * 10

    try:
        # Load original rawtlt data
        original_rawtlt_angles = np.loadtxt(ts.get_rawtlt_path())

        # Ensure bad_idx are within the range of im_data
        bad_idx = [idx for idx in bad_idx if idx < len(im_data)]

        # Create a mask of good tilts (True for keep, False for remove)
        mask = np.ones(len(im_data), dtype=bool)
        mask[bad_idx] = False

        # Apply mask to get cleaned data
        cleaned_mrc = im_data[mask]

        # Ensure the mask is the same length as original_rawtlt_angles
        if len(mask) > len(original_rawtlt_angles):
            mask = mask[: len(original_rawtlt_angles)]
        elif len(mask) < len(original_rawtlt_angles):
            mask = np.pad(
                mask,
                (0, len(original_rawtlt_angles) - len(mask)),
                "constant",
                constant_values=True,
            )

        # Apply mask to get cleaned rawtlt
        cleaned_ts_rawtlt = original_rawtlt_angles[mask]

        # Get paths to files
        original_ts_file = ts.get_mrc_path()
        original_ts_rawtlt = ts.get_rawtlt_path()

        # Backup original files
        logger.info(f"Backing up original tilt series to {original_ts_file}~")
        mrcfile.write(
            name=f"{original_ts_file}~", data=im_data, voxel_size=angpix, overwrite=True
        )

        logger.info(f"Backing up original rawtlt to {original_ts_rawtlt}~")
        np.savetxt(
            fname=f"{original_ts_rawtlt}~", X=original_rawtlt_angles, fmt="%0.2f"
        )

        # Write new data
        logger.info(f"Writing cleaned tilt series with {len(bad_idx)} tilts removed")
        mrcfile.write(
            name=f"{original_ts_file}",
            data=cleaned_mrc,
            voxel_size=angpix,
            overwrite=True,
        )

        logger.info(f"Writing cleaned rawtlt file")
        np.savetxt(fname=f"{original_ts_rawtlt}", X=cleaned_ts_rawtlt, fmt="%0.2f")

        # Update the TiltSeries object
        ts.tilt_angles = cleaned_ts_rawtlt
        ts.removed_indices = (
            sorted(set(ts.removed_indices + bad_idx))
            if hasattr(ts, "removed_indices") and ts.removed_indices
            else sorted(bad_idx)
        )

        logger.info(f"Successfully removed {len(bad_idx)} bad tilts")

    except Exception as e:
        logger.error(f"Error removing bad tilts: {e}")
        raise


def get_alignment_error(
    tilt_dir_name: Union[str, Path]
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Extract alignment error metrics from the alignment log file.

    Args:
        tilt_dir_name: Path to the directory containing alignment log

    Returns:
        Tuple of (known_unknown_ratio, residual_error, standard_deviation)
    """
    known_unknown_ratio = None
    resid_err = None
    sd = None

    log_path = Path(tilt_dir_name) / "align_patch.log"

    if not log_path.exists():
        logger.error(f"Alignment log file not found: {log_path}")
        return known_unknown_ratio, resid_err, sd

    try:
        with open(log_path, "r") as f_in:
            for line in f_in:
                if "Ratio of total measured values to all unknowns" in line:
                    known_unknown_ratio = float(line.split("=")[-1])
                if "Residual error mean and sd" in line:
                    parts = line.split()
                    resid_err = float(parts[5])
                    sd = parts[6]

        if known_unknown_ratio is None or resid_err is None or sd is None:
            logger.warning("Could not find all alignment statistics in the log file")

        return known_unknown_ratio, resid_err, sd

    except Exception as e:
        logger.error(f"Error reading alignment log: {e}")
        return None, None, None


def write_ta_coords_log(tilt_dir_name: Union[str, Path]) -> None:
    """
    Generate the taCoordinates.log file using alignlog command.

    Args:
        tilt_dir_name: Path to the tilt series directory
    """
    tilt_dir = Path(tilt_dir_name)
    log_file = tilt_dir / "taCoordinates.log"
    align_log = tilt_dir / "align_patch.log"

    if not align_log.exists():
        logger.error(f"Alignment log not found: {align_log}")
        return

    try:
        with open(log_file, "w") as ali_log:
            sbp_cmd = ["alignlog", "-c", str(align_log)]
            process = subprocess.run(
                sbp_cmd, stdout=ali_log, stderr=subprocess.PIPE, text=True
            )

            if process.returncode != 0:
                logger.error(f"Error generating taCoordinates.log: {process.stderr}")
            else:
                logger.info("Successfully generated taCoordinates.log")

    except Exception as e:
        logger.error(f"Error writing taCoordinates.log: {e}")


def improve_bad_alignments(tilt_dir_name: Union[str, Path], tilt_name: str) -> None:
    """
    Improve alignments by filtering out high-residual fiducials.

    Args:
        tilt_dir_name: Path to the tilt series directory
        tilt_name: Basename of the tilt series
    """
    tilt_dir = Path(tilt_dir_name)
    mod_file = tilt_dir / f"{tilt_name}.fid"
    mod2txt = tilt_dir / f"{tilt_name}.txt"
    txt4seed = tilt_dir / f"{tilt_name}.seed.txt"
    coords_log = tilt_dir / "taCoordinates.log"

    if not mod_file.exists():
        logger.error(f"Fiducial model file not found: {mod_file}")
        return

    if not coords_log.exists():
        logger.error(f"Coordinates log not found: {coords_log}")
        return

    try:
        # Load alignment log values
        align_log_vals = np.loadtxt(coords_log, skiprows=2)

        # Calculate median residual and find good points
        contour_resid, median_residual = calc.median_residual(align_log_vals)
        goodpoints = np.where(contour_resid[:, [1]] <= np.around(median_residual))
        goodpoints = goodpoints[0] + 1

        logger.info(
            f"Found {len(goodpoints)} good fiducials out of {len(contour_resid)}"
        )

        # Convert the fiducial model file to a text file
        subprocess.run(["model2point", "-contour", str(mod_file), str(mod2txt)])

        # Load fiducial text
        fid_text = np.loadtxt(mod2txt)

        # Create new contours using only good points
        new_good_contours = np.empty((0, 4))
        for i in range(goodpoints.shape[0]):
            a = fid_text[fid_text[:, 0] == goodpoints[i]]
            new_good_contours = np.append(new_good_contours, a, axis=0)

        # Renumber contours
        number_of_new_contours = np.unique(new_good_contours[:, 0]).shape[0]
        number_of_new_contours = np.arange(number_of_new_contours) + 1
        repeats = np.unique(new_good_contours[:, -1]).shape[0]
        new_contour_id = np.repeat(number_of_new_contours, repeats)
        new_contour_id = new_contour_id.reshape(new_contour_id.shape[0], 1)

        # Combine new IDs with position data
        new_good_contours = np.hstack((new_contour_id, new_good_contours[:, 1:]))

        # Save filtered contours
        np.savetxt(
            txt4seed, new_good_contours, fmt=" ".join(["%d"] + ["%0.3f"] * 2 + ["%d"])
        )

        # Convert filtered points back to model
        preali_path = tilt_dir / f"{tilt_name}_preali.mrc"
        if not preali_path.exists():
            logger.error(f"Prealigned stack not found: {preali_path}")
            return

        point_to_model_cmd = [
            "point2model",
            "-open",
            "-circle",
            "6",
            "-image",
            str(preali_path),
            str(txt4seed),
            str(mod_file),
        ]

        subprocess.run(point_to_model_cmd)
        logger.info("Created improved fiducial model with lower-residual points")

    except Exception as e:
        logger.error(f"Error improving alignments: {e}")


def detect_dark_tilts(
    ts_data: np.ndarray,
    ts_tilt_angles: np.ndarray,
    brightness_factor: float = 0.65,
    variance_factor: float = 0.1,
) -> List[int]:
    """
    Detect dark or low-quality tilts in the tilt series.

    Args:
        ts_data: Tilt series data array
        ts_tilt_angles: Array of tilt angles
        brightness_factor: Threshold factor for mean intensity (default: 0.65)
        variance_factor: Threshold factor for standard deviation (default: 0.1)

    Returns:
        List of indices of dark tilts
    """
    try:
        # Identify the reference image (typically at or near 0 tilt)
        tilt_series_mid_point_idx = calc.find_symmetric_tilt_reference(ts_tilt_angles)

        # Normalize images to 0-1 range
        normalized_images = calc.normalize(ts_data)
        reference_image = normalized_images[tilt_series_mid_point_idx, :, :]

        # Calculate statistics for the reference image
        reference_mean_intensity = np.mean(reference_image)
        reference_std_deviation = np.std(reference_image)

        # Calculate dynamic thresholds
        threshold_intensity = reference_mean_intensity * brightness_factor
        threshold_variance = reference_std_deviation * variance_factor

        # Calculate statistics for all tilts
        mean_intensities = np.mean(normalized_images, axis=(1, 2))
        std_deviations = np.std(normalized_images, axis=(1, 2))

        # Identify dark tilts based on mean intensity and standard deviation
        dark_frame_indices = np.where(
            (mean_intensities < threshold_intensity)
            & (std_deviations < threshold_variance)
        )[0].tolist()

        logger.info(f"Detected {len(dark_frame_indices)} dark tilts")
        return dark_frame_indices

    except Exception as e:
        logger.error(f"Error detecting dark tilts: {e}")
        return []


def swap_fast_slow_axes(tilt_dirname: Union[str, Path], tilt_name: str) -> None:
    """
    Swap the fast and slow axes of a tomogram.

    Args:
        tilt_dirname: Path to the tilt series directory
        tilt_name: Basename of the tilt series
    """
    rec_file = Path(tilt_dirname) / f"{tilt_name}.rec"

    if not rec_file.exists():
        logger.error(f"Reconstruction file not found: {rec_file}")
        return

    try:
        # Read the tomogram
        d, pixel_nm, _, _ = io.read_mrc(rec_file)

        # Swap axes
        d = np.swapaxes(d, 0, 1)

        # Write the reoriented tomogram
        mrcfile.write(
            rec_file,
            data=d,
            voxel_size=pixel_nm * 10,
            overwrite=True,
        )

        logger.info(f"Swapped fast and slow axes in {rec_file}")

    except Exception as e:
        logger.error(f"Error swapping axes: {e}")


def match_partial_filename(string_to_match: str, target_string: str) -> bool:
    """
    Check if a string matches part of another string.

    Args:
        string_to_match: String to look for
        target_string: String to search in

    Returns:
        True if string_to_match is in target_string, False otherwise
    """
    if string_to_match in target_string:
        return True
    else:
        logger.warning("Could not match string. Check if the file exists.")
        return False


def remove_xml_files(xml_file_path: Path) -> None:
    """
    Backup an XML file by renaming it.

    Args:
        xml_file_path: Path to the XML file
    """
    if xml_file_path.exists():
        backup_file_path = xml_file_path.with_suffix(".xml.bkp")
        shutil.move(xml_file_path, backup_file_path)
        logger.info(f"Backed up {xml_file_path} to {backup_file_path}")
    else:
        logger.warning(f"XML file {xml_file_path} not found.")
