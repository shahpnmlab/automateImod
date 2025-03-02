import mrcfile
import typer
import starfile
import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils
import automateImod.pio as pio
from automateImod.parallel import parallel_align_tilts

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Typer app
automateImod = typer.Typer(
    no_args_is_help=True, pretty_exceptions_show_locals=False, add_completion=False
)


def process_single_tilt_series(
    ts_path: Path,
    ts_mdoc_path: Optional[Path] = None,
    ts_tomostar_path: Optional[Path] = None,
    ts_tilt_axis: str = None,
    ts_bin: str = "1",
    ts_patch_size: str = None,
    min_fov: float = 0.7,
    max_attempts: int = 3,
    update_warp_xml: bool = False,
    reconstruct: bool = False,
    ts_xml_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Process a single tilt series: align, optionally update XML, and reconstruct.

    Args:
        ts_path: Path to the tilt series directory
        ts_mdoc_path: Path to the mdoc file directory
        ts_tomostar_path: Path to the tomostar file directory
        ts_tilt_axis: Tilt axis value
        ts_bin: Binning factor
        ts_patch_size: Patch size in Angstroms
        min_fov: Minimum required field of view
        max_attempts: Maximum alignment attempts
        update_warp_xml: Whether to update Warp XML files
        reconstruct: Whether to reconstruct tomograms
        ts_xml_path: Path to XML files (required if update_warp_xml is True)

    Returns:
        Dictionary with processing results
    """
    result = {"basename": ts_path.name, "success": False, "error": None}

    try:
        # Initialize tilt series object
        ts = pio.TiltSeries(
            path_to_ts_data=ts_path.parent,
            path_to_mdoc_data=ts_mdoc_path,
            path_to_tomostar=ts_tomostar_path,
            basename=ts_path.name,
            tilt_axis_ang=ts_tilt_axis,
            binval=ts_bin,
            patch_size_ang=ts_patch_size,
        )

        # Align tilt series
        align_result = align_tilt_series(ts, min_fov, max_attempts)
        result.update(align_result)

        # Update Warp XML if requested
        if update_warp_xml and ts_xml_path and result.get("success", False):
            xml_result = update_warp_xml_file(
                ts.basename,
                ts_xml_path,
                ts_tomostar_path,
                ts.tilt_dir_name,
            )
            result["xml_updated"] = xml_result

        # Reconstruct tomogram if requested
        if reconstruct and result.get("success", False):
            recon_result = reconstruct_tomogram(
                ts.tilt_dir_name,
                ts.basename,
                ts.extension,
                ts_bin,
            )
            result["reconstructed"] = recon_result

        return result

    except Exception as e:
        logger.error(f"Error processing tilt series {ts_path.name}: {str(e)}")
        result["error"] = str(e)
        return result


def align_tilt_series(
    ts: pio.TiltSeries, min_fov: float, max_attempts: int
) -> Dict[str, Any]:
    """
    Perform patch-based alignment of a tilt series.

    Args:
        ts: TiltSeries object
        min_fov: Minimum required field of view
        max_attempts: Maximum alignment attempts

    Returns:
        Dictionary with alignment results
    """
    result = {"success": False}

    ts_path = ts.get_mrc_path()
    marker_file = ts.tilt_dir_name / "autoImod.marker"

    if not ts_path.is_file():
        raise FileNotFoundError(f"No valid tilt series found at {ts_path}")

    # Read the tilt series data
    im_data = mrcfile.read(ts_path)

    # Generate command files
    coms.write_xcorr_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        tilt_extension=ts.extension,
        tilt_axis_ang=ts.tilt_axis_ang,
    )

    coms.make_xcorr_stack_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        tilt_extension=ts.extension,
        binval=ts.binval,
    )

    coms.make_patch_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        binval=ts.binval,
        tilt_axis_ang=ts.tilt_axis_ang,
        patch=ts.patch_size,
    )

    coms.track_patch_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        pixel_nm=ts.pixel_size / 10,  # Convert to nm
        binval=ts.binval,
        tilt_axis_ang=ts.tilt_axis_ang,
        dimX=ts.dimX,
        dimY=ts.dimY,
    )

    # Store original tilt frames and angles
    original_tilt_frames = ts.tilt_frames.copy()
    original_tilt_angles = ts.tilt_angles.copy()

    # Process tilt series if marker file doesn't exist
    if not marker_file.exists():
        result.update(process_initial_alignment(ts, im_data, marker_file, min_fov))

    # After all tilt removals, ensure the removed_indices are unique and sorted
    ts.removed_indices = sorted(set(ts.removed_indices))

    # Perform patch-based alignment
    logger.info(f"Performing patch-based alignment on {ts.basename}")
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/xcorr_patch.com", capture_output=False
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
    )

    # Write alignment coordinates log
    utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)

    # Get alignment error metrics
    known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)

    if resid_error is not None:
        logger.info(f"Residual error (nm): {resid_error}")
        result["residual_error"] = resid_error
        result["standard_deviation"] = sd
    else:
        logger.error(
            "Could not retrieve alignment statistics. The alignment may have failed."
        )
        return result

    # Attempt to improve alignments if necessary
    if (
        known_to_unknown is not None
        and resid_error is not None
        and known_to_unknown > 10
        and resid_error >= 1.5
    ):
        result.update(
            improve_alignments(ts, known_to_unknown, resid_error, max_attempts)
        )

    # Log final alignment accuracy
    if resid_error is not None:
        logger.info(f"The final alignment accuracy is:")
        logger.info(f"Residual error (nm): {resid_error} (SD: {sd})")
        result["final_residual_error"] = resid_error
        result["final_standard_deviation"] = sd
        result["success"] = True
    else:
        logger.error("Could not retrieve final alignment statistics.")

    return result


def process_initial_alignment(
    ts: pio.TiltSeries, im_data: np.ndarray, marker_file: Path, min_fov: float
) -> Dict[str, Any]:
    """
    Process the initial alignment steps, detecting and removing dark tilts and large shifts.

    Args:
        ts: TiltSeries object
        im_data: Image data array
        marker_file: Path to the marker file
        min_fov: Minimum required field of view

    Returns:
        Dictionary with processing results
    """
    result = {}

    # Create marker file
    marker_file.touch()
    logger.info(f"Marker file not detected in {marker_file.parent}\nProcessing...")

    # Detect and remove dark tilts
    logger.info("Looking for dark tilts...")
    dark_frame_indices = utils.detect_dark_tilts(
        ts_data=im_data, ts_tilt_angles=ts.tilt_angles
    )

    if dark_frame_indices:
        logger.info(f"Detected {len(dark_frame_indices)} dark tilts in {ts.basename}")
        logger.info(f"Removing dark tilts...")
        utils.remove_bad_tilts(
            ts=ts,
            im_data=im_data,
            pixel_nm=ts.pixel_size / 10,
            bad_idx=dark_frame_indices,
        )
        ts.removed_indices.extend(dark_frame_indices)

        # Reload data after removing dark tilts
        del im_data
        im_data, _, _, _ = pio.read_mrc(ts.get_mrc_path())

        result["dark_tilts_removed"] = len(dark_frame_indices)
    else:
        logger.info(
            "No dark frames found in the current tilt series. Proceeding with coarse alignments..."
        )

    # Write marker file for dark frames
    with open(marker_file, "w") as fout:
        fout.write("frame_basename,stage_angle,pos_in_tilt_stack\n")
        for idx in dark_frame_indices:
            if idx < len(ts.tilt_frames) and idx < len(ts.tilt_angles):
                frame_name = ts.tilt_frames[idx]
                fout.write(f"{frame_name},{ts.tilt_angles[idx]},{idx}\n")

    # Run coarse alignment
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/xcorr_coarse.com", capture_output=False
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/newst_coarse.com", capture_output=False
    )

    # Detect and remove large shifts
    large_shift_indices = utils.detect_large_shifts_afterxcorr(
        coarse_align_prexg=f"{ts.tilt_dir_name}/{ts.basename}.prexg",
        pixel_size_nm=ts.pixel_size / 10,
        image_size=(ts.dimX, ts.dimY),
        min_fov_fraction=min_fov,
    )

    if large_shift_indices:
        logger.info(
            f"Detected {len(large_shift_indices)} badly tracking tilts in {ts.basename}"
        )
        logger.info(f"Removing badly tracked tilts...")

        # Map indices back to original stack indices
        original_tilt_frames = ts.tilt_frames.copy()
        original_tilt_angles = ts.tilt_angles.copy()

        # Convert decimated stack indices to original stack indices
        original_large_shift_indices = []
        for idx in large_shift_indices:
            if idx < len(ts.tilt_angles):
                matches = np.where(
                    np.array(original_tilt_angles) == ts.tilt_angles[idx]
                )[0]
                if len(matches) > 0:
                    original_large_shift_indices.append(matches[0])

        # Remove bad tilts
        utils.remove_bad_tilts(
            ts=ts,
            im_data=im_data,
            pixel_nm=ts.pixel_size / 10,
            bad_idx=large_shift_indices,
        )
        ts.removed_indices.extend(original_large_shift_indices)

        logger.info(f"Redoing coarse alignment with decimated {ts.basename} stack")
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/xcorr_coarse.com", capture_output=False
        )
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/newst_coarse.com", capture_output=False
        )

        # Append large shift frames to marker file
        with open(marker_file, "a") as fout:
            for idx in original_large_shift_indices:
                if idx < len(original_tilt_angles) and idx < len(original_tilt_frames):
                    frame_name = Path(original_tilt_frames[idx]).stem
                    fout.write(f"{frame_name},{original_tilt_angles[idx]},{idx}\n")
                else:
                    logger.warning(
                        f"Warning: Index {idx} is out of range for original tilt_angles or tilt_frames."
                    )

        result["large_shifts_removed"] = len(large_shift_indices)

    return result


def improve_alignments(
    ts: pio.TiltSeries, known_to_unknown: float, resid_error: float, max_attempts: int
) -> Dict[str, Any]:
    """
    Attempt to improve alignments when the initial alignment is poor.

    Args:
        ts: TiltSeries object
        known_to_unknown: Ratio of known to unknown values
        resid_error: Residual error in nm
        max_attempts: Maximum number of improvement attempts

    Returns:
        Dictionary with improvement results
    """
    result = {"improvement_attempts": 0}

    logger.info(f"The alignment statistics for {ts.basename} are worse than expected.")
    logger.info(f"Will try to improve the alignments in {max_attempts} attempts.")

    attempt = 0
    initial_resid_error = resid_error

    while attempt < max_attempts:
        logger.info(f"Attempt: {attempt + 1}")

        utils.improve_bad_alignments(
            tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename
        )

        logger.info(f"Realigning {ts.tilt_dir_name} with a new model as seed.")
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
        )

        utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)
        attempt += 1

        known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)

        if resid_error is not None:
            logger.info(f"Residual error (nm): {resid_error}")
            result[f"attempt_{attempt}_residual"] = resid_error
        else:
            logger.warning("Could not retrieve alignment statistics.")

        if not (
            known_to_unknown is not None
            and resid_error is not None
            and known_to_unknown > 10
            and resid_error >= 1.5
        ):
            break

    result["improvement_attempts"] = attempt

    if attempt == max_attempts:
        logger.warning("Maximum number of realignment attempts reached.")
        result["max_attempts_reached"] = True

    if initial_resid_error and resid_error:
        result["improvement_percentage"] = (
            (initial_resid_error - resid_error) / initial_resid_error
        ) * 100

    return result


def update_warp_xml_file(
    ts_basename: str,
    ts_xml_path: Path,
    ts_tomostar_path: Path,
    ts_log_path: Path,
) -> bool:
    """
    Update Warp XML and tomostar files based on identified bad frames.

    Args:
        ts_basename: Basename of the tilt series
        ts_xml_path: Path to XML files
        ts_tomostar_path: Path to tomostar files
        ts_log_path: Path to log files

    Returns:
        True if successful, False otherwise
    """
    import xml.etree.ElementTree as ET
    import pandas as pd

    logger.info("Updating the provided XML metadata file and tomostar file")

    # Read the autoImod.marker file
    marker_file = Path(ts_log_path) / "autoImod.marker"
    if not marker_file.exists():
        logger.warning(f"Warning: {marker_file} does not exist.")
        return False

    md = pd.read_csv(marker_file, delimiter=",")
    bad_frames = md["frame_basename"].tolist()

    # Update XML file
    xml_file = Path(ts_xml_path) / f"{ts_basename}.xml"
    if not xml_file.exists():
        logger.error(f"Error: XML file {xml_file} does not exist.")
        return False

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        elements_to_update = [
            "UseTilt",
            "AxisAngle",
            "Angles",
            "AxisOffsetX",
            "AxisOffsetY",
            "MoviePath",
        ]

        for element_name in elements_to_update:
            element = root.find(element_name)
            if element is not None:
                lines = element.text.strip().split("\n")
                updated_lines = []

                for line, movie in zip(
                    lines, root.find("MoviePath").text.strip().split("\n")
                ):
                    frame_name = Path(movie).stem
                    if frame_name not in bad_frames:
                        updated_lines.append(line)

                element.text = (
                    "\n" + "\n".join(updated_lines) + "\n"
                )  # Preserve formatting
            else:
                logger.warning(
                    f"Warning: {element_name} element not found in the XML file."
                )

        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
        logger.info(f"Updated XML file: {xml_file}")

        # Update tomostar file
        tomostar_file = Path(ts_tomostar_path) / f"{ts_basename}.tomostar"
        if not tomostar_file.exists():
            logger.error(f"Error: Tomostar file {tomostar_file} does not exist.")
            return False

        tomostar_data = starfile.read(tomostar_file)

        # Check if tomostar_data is a Series or DataFrame
        if isinstance(tomostar_data["wrpMovieName"], pd.Series):
            # If it's a Series, convert it to a DataFrame
            tomostar_data = pd.DataFrame(tomostar_data)

        # Filter out bad frames
        good_mask = (
            ~tomostar_data["wrpMovieName"]
            .apply(lambda x: Path(x).stem)
            .isin(bad_frames)
        )
        updated_tomostar_data = tomostar_data[good_mask]

        starfile.write(updated_tomostar_data, tomostar_file, overwrite=True)
        logger.info(f"Updated tomostar file: {tomostar_file}")

        return True

    except Exception as e:
        logger.error(f"Error updating XML and tomostar files: {str(e)}")
        return False


def reconstruct_tomogram(
    ts_data_path: Path,
    ts_basename: str,
    ts_extension: str,
    tomo_bin: str,
) -> bool:
    """
    Reconstruct a tomogram from an aligned tilt series.

    Args:
        ts_data_path: Path to tilt series data
        ts_basename: Basename of the tilt series
        ts_extension: File extension of the tilt series
        tomo_bin: Binning factor for tomogram reconstruction

    Returns:
        True if successful, False otherwise
    """
    try:
        tomo = pio.Tomogram(
            path_to_data=ts_data_path,
            basename=ts_basename,
            extension=ts_extension,
            binval=tomo_bin,
        )

        _, pixel_nm, dimX, dimY = pio.read_mrc(
            f"{tomo.tilt_dir_name}/{tomo.basename}.{tomo.extension}"
        )

        slab_thickness = calc.get_thickness(
            unbinned_voxel_size=pixel_nm * 10, binval=tomo.binval
        )

        coms.make_tomogram(
            tilt_dir_name=tomo.tilt_dir_name,
            tilt_name=tomo.basename,
            tilt_extension=tomo.extension,
            binval=tomo.binval,
            dimX=dimX,
            dimY=dimY,
            thickness=slab_thickness,
        )

        coms.execute_com_file(
            f"{str(tomo.tilt_dir_name)}/newst_ali.com",
        )

        coms.execute_com_file(f"{str(tomo.tilt_dir_name)}/tilt_ali.com")

        utils.swap_fast_slow_axes(tomo.tilt_dir_name, tomo.basename)

        logger.info(f"Successfully reconstructed tomogram: {ts_basename}")
        return True

    except Exception as e:
        logger.error(f"Error reconstructing tomogram: {str(e)}")
        return False


@automateImod.command(no_args_is_help=True)
def align_tilts(
    ts_basename: str = typer.Option(..., help="Tilt series basename e.g. Position_1"),
    ts_data_path: Path = typer.Option(
        ..., help="Directory containing tilt series data"
    ),
    ts_mdoc_path: Optional[Path] = typer.Option(
        None, help="Directory containing the tilt series mdoc file"
    ),
    ts_tomostar_path: Optional[Path] = typer.Option(
        None, help="Directory containing the tomostar file"
    ),
    ts_xml_path: Optional[Path] = typer.Option(
        None, help="Directory containing the XML file (required for --update-warp-xml)"
    ),
    ts_tilt_axis: str = typer.Option(..., help="Tilt axis value"),
    ts_bin: str = typer.Option("1", help="Bin value to reduce the tilt series size by"),
    ts_patch_size: str = typer.Option(
        ..., help="Size of patches to perform patch_tracking (Å)"
    ),
    min_fov: float = typer.Option(0.7, help="Minimum required field of view"),
    max_attempts: int = typer.Option(
        3, help="How many attempts before quitting refinement"
    ),
    n_cpu: Optional[int] = typer.Option(
        None, help="Number of CPU cores to use for parallel processing"
    ),
    pattern: str = typer.Option(
        "*", help="Glob pattern to match tilt series basenames"
    ),
    update_warp_xml: bool = typer.Option(
        False, help="Update Warp XML files after alignment"
    ),
    reconstruct: bool = typer.Option(
        False, help="Reconstruct tomograms after alignment"
    ),
):
    """
    Perform patch-based tilt series tracking using IMOD routines.
    Optionally update Warp XML files and reconstruct tomograms.
    """
    # Set up logging
    utils.setup_logger()

    # Check if XML path is provided when update_warp_xml is True
    if update_warp_xml and not ts_xml_path:
        logger.error(
            "XML path (--ts-xml-path) is required when --update-warp-xml is used"
        )
        raise typer.Abort()

    # Determine if we're processing a single tilt series or multiple
    if "*" in ts_basename or "?" in ts_basename or pattern != "*":
        # Process multiple tilt series in parallel
        logger.info(f"Processing multiple tilt series matching pattern '{pattern}'")

        # Use parallel processing module to handle multiple tilt series
        results = parallel_align_tilts(
            data_path=ts_data_path,
            pattern=pattern,
            n_cpu=n_cpu,
            ts_mdoc_path=ts_mdoc_path,
            ts_tomostar_path=ts_tomostar_path,
            ts_tilt_axis=ts_tilt_axis,
            ts_bin=ts_bin,
            ts_patch_size=ts_patch_size,
            min_fov=min_fov,
            max_attempts=max_attempts,
            update_warp_xml=update_warp_xml,
            reconstruct=reconstruct,
            ts_xml_path=ts_xml_path,
        )

        # Log summary of results
        success_count = sum(1 for r in results if r.get("success", False))
        logger.info(f"Processed {len(results)} tilt series, {success_count} successful")

    else:
        # Process a single tilt series
        logger.info(f"Processing single tilt series: {ts_basename}")

        result = process_single_tilt_series(
            Path(ts_data_path) / ts_basename,
            ts_mdoc_path=ts_mdoc_path,
            ts_tomostar_path=ts_tomostar_path,
            ts_tilt_axis=ts_tilt_axis,
            ts_bin=ts_bin,
            ts_patch_size=ts_patch_size,
            min_fov=min_fov,
            max_attempts=max_attempts,
            update_warp_xml=update_warp_xml,
            reconstruct=reconstruct,
            ts_xml_path=ts_xml_path,
        )

        if result.get("success", False):
            logger.info(f"Successfully processed tilt series: {ts_basename}")
        else:
            logger.error(f"Failed to process tilt series: {ts_basename}")
            if result.get("error"):
                logger.error(f"Error: {result['error']}")


@automateImod.command()
def update_warp_xml(
    ts_basename: str = typer.Option(..., help="Basename of the tilt series"),
    ts_xml_path: Path = typer.Option(..., help="Path to Warp processing results"),
    ts_tomostar_path: Path = typer.Option(..., help="Path to tomostar file"),
    ts_log_path: Path = typer.Option(..., help="Path to imod processing directory"),
):
    """
    Read in the log file generated by align-tilts, remove tilts with large shifts from the XML file,
    and update the tomostar file.

    Note: This functionality is now available in align-tilts command with --update-warp-xml flag.
    """
    utils.setup_logger()
    logger.info("Starting XML update process")

    success = update_warp_xml_file(
        ts_basename=ts_basename,
        ts_xml_path=ts_xml_path,
        ts_tomostar_path=ts_tomostar_path,
        ts_log_path=ts_log_path,
    )

    if success:
        logger.info("Successfully updated XML and tomostar files")
    else:
        logger.error("Failed to update XML and tomostar files")


@automateImod.command()
def reconstruct_tomograms(
    ts_data_path: Path = typer.Option(
        ..., help="Directory containing tilt series data"
    ),
    ts_basename: str = typer.Option(..., help="Tilt series basename e.g. Position_"),
    ts_extension: str = typer.Option(
        "mrc", help="Does the TS end with an st or mrc extension?"
    ),
    tomo_bin: str = typer.Option(..., help="Binned tomogram size"),
    n_cpu: Optional[int] = typer.Option(
        None, help="Number of CPU cores to use for parallel processing"
    ),
    pattern: str = typer.Option(
        "*", help="Glob pattern to match tilt series basenames"
    ),
):
    """
    Reconstruct tomograms using IMOD of aligned tilt series for quick visualization.

    Note: This functionality is now available in align-tilts command with --reconstruct flag.
    """
    utils.setup_logger()

    # Determine if we're processing a single tomogram or multiple
    if "*" in ts_basename or "?" in ts_basename or pattern != "*":
        from automateImod.parallel import find_tilt_series, parallel_process

        # Find all tilt series matching the pattern
        tilt_series_list = find_tilt_series(ts_data_path, pattern)

        if not tilt_series_list:
            logger.warning(
                f"No tilt series found matching pattern '{pattern}' in {ts_data_path}"
            )
            return

        logger.info(f"Found {len(tilt_series_list)} tilt series to reconstruct")

        # Process tomograms in parallel
        results = parallel_process(
            tilt_series_list,
            lambda path: reconstruct_tomogram(
                ts_data_path, path.name, ts_extension, tomo_bin
            ),
            n_workers=n_cpu,
        )

        success_count = sum(1 for r in results if r)
        logger.info(f"Reconstructed {success_count} of {len(results)} tomograms")

    else:
        # Process a single tomogram
        logger.info(f"Reconstructing single tomogram: {ts_basename}")

        success = reconstruct_tomogram(
            ts_data_path, ts_basename, ts_extension, tomo_bin
        )

        if success:
            logger.info(f"Successfully reconstructed tomogram: {ts_basename}")
        else:
            logger.error(f"Failed to reconstruct tomogram: {ts_basename}")


if __name__ == "__main__":
    automateImod()
