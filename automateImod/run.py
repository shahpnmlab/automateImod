import math
import mrcfile
import typer
import starfile
import shutil
from typing import Optional
from tqdm import tqdm
import os
import importlib.metadata

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils
import automateImod.pio as pio
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from dask.distributed import Client, LocalCluster, as_completed

automateimod = typer.Typer(
    no_args_is_help=True, pretty_exceptions_show_locals=False, add_completion=False
)


def _version_callback(value: bool):
    if value:
        version = importlib.metadata.version("automateImod")
        typer.echo(f"automateImod {version}")
        raise typer.Exit()


@automateimod.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    pass


def task_setup(
    basename,
    ts_data_folder,
    ts_mdoc_path,
    ts_tomostar_path,
    ts_xml_path,
    ts_tilt_axis,
    ts_bin,
    ts_patch_size,
):
    """Initial setup task: create TiltSeries object and logger."""
    log_path = ts_data_folder / basename / f"{basename}.log"
    logger = utils.setup_logging(log_path)
    logger.info(f"Starting processing for tilt series: {basename}")
    ts = pio.TiltSeries(
        path_to_ts_data=ts_data_folder,
        path_to_mdoc_data=ts_mdoc_path,
        path_to_tomostar=ts_tomostar_path,
        basename=basename,
        tilt_axis_ang=ts_tilt_axis,
        binval=ts_bin,
        patch_size_ang=ts_patch_size,
        logger=logger,
    )

    # If tilt_frames is empty, try to read from XML file as fallback
    if not ts.tilt_frames and ts_xml_path:
        xml_file = ts_xml_path / f"{basename}.xml"
        if xml_file.exists():
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                movie_path_element = root.find("MoviePath")
                if movie_path_element is not None:
                    movie_paths = movie_path_element.text.strip().split("\n")
                    ts.tilt_frames = [Path(movie).name for movie in movie_paths]
                    logger.info(
                        f"Loaded {len(ts.tilt_frames)} tilt frames from XML file."
                    )
                else:
                    logger.warning("MoviePath element not found in XML file.")
            except Exception as e:
                logger.error(f"Error reading frame names from XML file: {e}")
        else:
            logger.warning(
                f"XML file {xml_file} does not exist, cannot load frame names."
            )

    # Final check: if tilt_frames is still empty, try reading from tomostar directly
    if not ts.tilt_frames and ts_tomostar_path:
        tomostar_file = ts_tomostar_path / f"{basename}.tomostar"
        if tomostar_file.exists():
            try:
                tomostar_data = starfile.read(tomostar_file)
                ts.tilt_frames = [
                    Path(movie_name).name
                    for movie_name in tomostar_data["wrpMovieName"]
                ]
                logger.info(
                    f"Loaded {len(ts.tilt_frames)} tilt frames from tomostar file (fallback)."
                )
            except Exception as e:
                logger.error(f"Error reading frame names from tomostar file: {e}")

    if not ts.tilt_frames:
        logger.error(
            f"Could not load frame names for {basename}. "
            f"Please provide either --ts-tomostar-path, --ts-mdoc-path, or --ts-xml-path."
        )

    # Validate that all metadata sources agree on frame count before any processing
    utils.validate_tilt_series_counts(ts, ts_xml_path, logger)

    return ts, logger


def task_preprocessing(setup_result, ts_intensity_threshold):
    """Task for preprocessing: bad tilt detection and removal.

    Detection strategy (in priority order):
      1. If tomostar intensity/dose data are available AND ts_intensity_threshold
         is set, use filter_by_intensities.
      2. Otherwise fall back to pixel-level dark-tilt detection.
    All bad indices are collected and remove_bad_tilts is called exactly once.
    """
    ts, logger = setup_result
    ts_path = ts.get_mrc_path()
    if not ts_path.is_file():
        logger.error(f"No valid tilt series found at {ts_path}. Aborting.")
        return None

    im_data = mrcfile.read(ts_path)
    original_tilt_frames = ts.tilt_frames.copy()
    original_tilt_angles = ts.tilt_angles.copy()
    current_to_original_idx = list(range(len(original_tilt_frames)))

    # --- Choose detection strategy ---
    has_tomostar_data = len(ts.doses) > 0 and len(ts.intensities) > 0

    if ts_intensity_threshold is not None and has_tomostar_data:
        logger.info(
            f"Aligning {ts.basename}: Detecting bad tilts via tomostar intensity "
            f"(threshold={ts_intensity_threshold})"
        )
        bad_frame_indices = utils.filter_by_intensities(
            ts=ts, threshold=ts_intensity_threshold, logger=logger
        )
    else:
        if ts_intensity_threshold is not None and not has_tomostar_data:
            logger.warning(
                "Intensity threshold was set but tomostar dose/intensity data are "
                "unavailable. Falling back to pixel-level dark tilt detection."
            )
        else:
            logger.info(f"Aligning {ts.basename}: Detecting dark frames")
        bad_frame_indices = utils.detect_dark_tilts(
            ts_data=im_data, ts_tilt_angles=ts.tilt_angles, logger=logger
        )

    # --- Remove bad frames (single call) ---
    if len(bad_frame_indices) > 0:
        logger.info(
            f"Detected {len(bad_frame_indices)} bad tilts in {ts.basename}; "
            f"removing and rebuilding stack."
        )
        utils.remove_bad_tilts(
            ts=ts,
            im_data=im_data,
            pixel_nm=ts.pixel_size / 10,
            bad_idx=bad_frame_indices,
        )
        ts.removed_indices.extend(bad_frame_indices)

        bad_frame_indices_set = set(bad_frame_indices)
        current_to_original_idx = [
            idx for idx in current_to_original_idx if idx not in bad_frame_indices_set
        ]
        logger.info(
            f"After removal, current stack has {len(current_to_original_idx)} frames."
        )

        del im_data
        im_data, _, _, _ = pio.read_mrc(ts_path)
    else:
        logger.info("No bad frames detected.")

    # --- Write marker file ---
    marker_file = ts.tilt_dir_name / "autoImod.marker"
    if not marker_file.exists():
        marker_file.touch()
        with open(marker_file, "w") as fout:
            fout.write("frame_basename,stage_angle,pos_in_tilt_stack\n")
            for idx in bad_frame_indices:
                if idx < len(original_tilt_frames) and idx < len(original_tilt_angles):
                    fout.write(
                        f"{original_tilt_frames[idx]},"
                        f"{original_tilt_angles[idx]},"
                        f"{idx}\n"
                    )

    return (
        ts,
        im_data,
        original_tilt_angles,
        original_tilt_frames,
        current_to_original_idx,
        logger,
    )


def task_coarse_alignment(
    preprocessing_result,
    min_fov,
):
    """Task for coarse alignment and large shift detection."""
    if preprocessing_result is None:
        return None
    (
        ts,
        im_data,
        original_tilt_angles,
        original_tilt_frames,
        current_to_original_idx,
        logger,
    ) = preprocessing_result
    logger.info(f"Aligning {ts.basename}: Running coarse alignment")
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
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/xcorr_coarse.com", capture_output=False, logger=logger
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/newst_coarse.com", capture_output=False, logger=logger
    )

    large_shift_indices = utils.detect_large_shifts_afterxcorr(
        coarse_align_prexg=f"{ts.tilt_dir_name}/{ts.basename}.prexg",
        image_size=(ts.dimX, ts.dimY),
        logger=logger,
        min_fov_fraction=min_fov,
    )

    if len(large_shift_indices) > 0:
        logger.info(
            f"Detected {len(large_shift_indices)} badly tracking tilts in {ts.basename}"
        )
        logger.info(f"Aligning {ts.basename}: Removing bad tilts and rebuilding stack")

        original_large_shift_indices = []
        for idx in large_shift_indices:
            if idx < len(current_to_original_idx):
                original_idx = current_to_original_idx[idx]
                original_large_shift_indices.append(original_idx)
                logger.info(
                    f"Current index {idx} maps to original index {original_idx}"
                )
            else:
                logger.error(
                    f"Current index {idx} is out of range (current stack has "
                    f"{len(current_to_original_idx)} frames)"
                )

        utils.remove_bad_tilts(
            ts=ts,
            im_data=im_data,
            pixel_nm=ts.pixel_size / 10,
            bad_idx=large_shift_indices,
        )
        ts.removed_indices.extend(original_large_shift_indices)
        logger.info(f"Redoing coarse alignment with decimated {ts.basename} stack")

        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/xcorr_coarse.com",
            capture_output=False,
            logger=logger,
        )
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/newst_coarse.com",
            capture_output=False,
            logger=logger,
        )
        marker_file = ts.tilt_dir_name / "autoImod.marker"
        with open(marker_file, "a") as fout:
            for original_idx in original_large_shift_indices:
                if original_idx < len(original_tilt_angles) and original_idx < len(
                    original_tilt_frames
                ):
                    frame_name = original_tilt_frames[original_idx]
                    fout.write(
                        f"{frame_name},{original_tilt_angles[original_idx]},{original_idx}\n"
                    )
                else:
                    logger.error(
                        f"Original index {original_idx} is out of range for "
                        f"original tilt_angles or tilt_frames."
                    )
    return ts, logger


def task_fine_alignment(coarse_align_result, max_attempts):
    """Task for fine alignment and alignment improvement."""
    if coarse_align_result is None:
        return None
    ts, logger = coarse_align_result
    logger.info(f"Aligning {ts.basename}: Running fine alignment")
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
        pixel_nm=ts.pixel_size / 10,
        binval=ts.binval,
        tilt_axis_ang=ts.tilt_axis_ang,
        dimX=ts.dimX,
        dimY=ts.dimY,
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/xcorr_patch.com", capture_output=False, logger=logger
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False, logger=logger
    )
    utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name, logger=logger)

    initial_contours, initial_points = utils.get_alignment_track_stats(ts.tilt_dir_name)
    if initial_contours is not None and initial_contours > 0:
        min_contour_threshold = max(8, int(math.ceil(initial_contours * 0.5)))
    else:
        min_contour_threshold = None

    known_to_unknown, resid_error, sd = utils.get_alignment_error(
        ts.tilt_dir_name, logger
    )

    if resid_error is None:
        logger.warning("Alignment failed, could not retrieve statistics.")
        return ts, logger

    logger.info(f"Residual error (nm): {resid_error}")

    if known_to_unknown > 10 and resid_error >= 1.5:
        contours_floor_triggered = False
        logger.warning(f"Alignment for {ts.basename} is worse than expected.")
        for attempt in range(max_attempts):
            logger.info(
                f"Aligning {ts.basename}: Improving alignment (attempt {attempt + 1})"
            )
            utils.improve_bad_alignments(
                tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, logger=logger
            )
            coms.execute_com_file(
                f"{str(ts.tilt_dir_name)}/align_patch.com",
                capture_output=False,
                logger=logger,
            )
            utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name, logger=logger)
            current_contours, current_points = utils.get_alignment_track_stats(
                ts.tilt_dir_name
            )
            if (
                min_contour_threshold is not None
                and current_contours is not None
                and current_contours < min_contour_threshold
            ):
                logger.warning(
                    "Stopping alignment improvement early: tracked contours dropped from "
                    f"{initial_contours} to {current_contours}, which is below the "
                    f"safeguard threshold of {min_contour_threshold}."
                )
                contours_floor_triggered = True
                break
            _, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name, logger)
            if resid_error is not None and resid_error < 1.5:
                logger.info(f"Alignment improved. Residual error: {resid_error} nm")
                break
        else:
            logger.warning("Max realignment attempts reached.")
        if contours_floor_triggered:
            logger.warning(
                "Alignment improvement aborted because too few contours remain "
                "to provide reliable statistics."
            )

    logger.info(
        f"Final alignment accuracy: Residual error (nm): {resid_error} (SD: {sd})"
    )
    return ts, logger


def task_final_stack(fine_align_result):
    """Task for building the final aligned stack."""
    if fine_align_result is None:
        return None
    ts, logger = fine_align_result
    logger.info(f"Building aligned stack for {ts.basename}...")
    coms.make_ali_stack_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        tilt_extension=ts.extension,
        binval=ts.binval,
        dimX=ts.dimX,
        dimY=ts.dimY,
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/newst_ali.com", capture_output=False, logger=logger
    )
    return ts, logger


def task_update_warp_xml(final_stack_result, ts_xml_path, ts_tomostar_path, backup_xml):
    """Task for updating Warp XML and TomoStar files."""
    if final_stack_result is None:
        return
    ts, logger = final_stack_result
    if not (ts_xml_path or ts_tomostar_path):
        logger.warning(
            "Skipping Warp XML/tomostar update because neither path is provided."
        )
        return

    logger.info("Updating the provided XML metadata file and/or tomostar file")
    marker_file = ts.tilt_dir_name / "autoImod.marker"
    if not marker_file.exists():
        logger.warning(f"{marker_file} does not exist.")
        return

    md = pd.read_csv(marker_file, delimiter=",")
    bad_frames = set(md["frame_basename"].tolist())

    if len(bad_frames) == 0:
        logger.info("No frames to remove, skipping XML/tomostar update.")
        return

    logger.info(f"Removing {len(bad_frames)} frames: {bad_frames}")

    actual_final_count = None

    if ts_xml_path:
        xml_file = ts_xml_path / f"{ts.basename}.xml"
        if not xml_file.exists():
            logger.error(f"XML file {xml_file} does not exist.")
        else:
            if backup_xml:
                backup_file_path = xml_file.with_suffix(xml_file.suffix + ".bak")
                try:
                    shutil.copy2(xml_file, backup_file_path)
                    logger.info(f"Backup of {xml_file} created at {backup_file_path}")
                except Exception as e:
                    logger.error(f"Error creating backup for {xml_file}: {e}")

            tree = ET.parse(xml_file)
            root = tree.getroot()

            movie_path_element = root.find("MoviePath")
            if movie_path_element is None:
                logger.error("MoviePath element not found in XML file.")
            else:
                movie_paths = movie_path_element.text.strip().split("\n")
                original_count = len(movie_paths)
                logger.info(f"Original XML has {original_count} views")

                good_indices = []
                removed_count = 0
                for i, movie in enumerate(movie_paths):
                    frame_name = Path(movie).name
                    if frame_name not in bad_frames:
                        good_indices.append(i)
                    else:
                        removed_count += 1
                        logger.info(f"Removing view {i}: {frame_name}")

                expected_final_count = original_count - len(bad_frames)
                actual_final_count = len(good_indices)

                if actual_final_count != expected_final_count:
                    logger.warning(
                        f"Mismatch: expected {expected_final_count} views after removal, "
                        f"but got {actual_final_count}. Some frames in marker file may "
                        f"not exist in XML."
                    )

                logger.info(
                    f"Keeping {actual_final_count} views, removed {removed_count} views"
                )

                elements_to_update = [
                    "Angles",
                    "Dose",
                    "UseTilt",
                    "AxisAngle",
                    "AxisOffsetX",
                    "AxisOffsetY",
                    "MoviePath",
                ]

                for element_name in elements_to_update:
                    element = root.find(element_name)
                    if element is not None:
                        lines = element.text.strip().split("\n")
                        if len(lines) != original_count:
                            logger.warning(
                                f"{element_name} has {len(lines)} entries but "
                                f"MoviePath has {original_count}. "
                                f"This may cause misalignment."
                            )
                        updated_lines = [
                            lines[i] for i in good_indices if i < len(lines)
                        ]
                        element.text = "\n" + "\n".join(updated_lines) + "\n"
                        logger.info(
                            f"Updated {element_name}: {len(lines)} -> "
                            f"{len(updated_lines)} entries"
                        )
                    else:
                        logger.warning(
                            f"{element_name} element not found in the XML file."
                        )

                tree.write(xml_file, encoding="utf-8", xml_declaration=True)
                logger.info(f"Updated XML file: {xml_file}")

    if ts_tomostar_path:
        tomostar_file = ts_tomostar_path / f"{ts.basename}.tomostar"
        tomostar_file_bkp = ts_tomostar_path / f"{ts.basename}.tomostar.bkp"
        if not tomostar_file.exists():
            logger.error(f"Tomostar file {tomostar_file} does not exist.")
        else:
            shutil.copy2(tomostar_file, tomostar_file_bkp)
            tomostar_data = starfile.read(tomostar_file)
            original_tomostar_count = len(tomostar_data)
            logger.info(f"Original tomostar has {original_tomostar_count} entries")

            good_mask = (
                ~tomostar_data["wrpMovieName"]
                .apply(lambda x: Path(x).name)
                .isin(bad_frames)
            )
            updated_tomostar_data = tomostar_data[good_mask]
            final_tomostar_count = len(updated_tomostar_data)

            logger.info(
                f"Tomostar: {original_tomostar_count} -> {final_tomostar_count} entries"
            )

            if (
                actual_final_count is not None
                and final_tomostar_count != actual_final_count
            ):
                logger.warning(
                    f"Mismatch between XML ({actual_final_count}) and tomostar "
                    f"({final_tomostar_count}) final counts. Check for inconsistencies."
                )

            starfile.write(updated_tomostar_data, tomostar_file, overwrite=True)
            logger.info(f"Updated tomostar file: {tomostar_file}")


def task_reconstruct_tomo(final_stack_result, tomo_bin):
    """Task for reconstructing a tomogram."""
    if final_stack_result is None:
        return
    ts, logger = final_stack_result
    if not tomo_bin:
        logger.warning("Skipping reconstruction because tomo_bin is not provided.")
        return

    logger.info(f"Reconstructing tomogram for {ts.basename}")
    tomo = pio.Tomogram(
        path_to_data=ts.tilt_dir_name.parent,
        basename=ts.basename,
        extension=ts.extension,
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
        f"{str(tomo.tilt_dir_name)}/tilt_ali.com", capture_output=False, logger=logger
    )
    utils.swap_fast_slow_axes(tomo.tilt_dir_name, tomo.basename)


@automateimod.command(no_args_is_help=True)
def align_tilts(
    ts_data_folder: Path = typer.Option(
        ...,
        "--ts-data-folder",
        help="Path to the folder containing tilt series subdirectories.",
    ),
    ts_basenames: Optional[str] = typer.Option(
        None,
        "--ts-basenames",
        help="A comma-separated string of specific tilt series basenames to process.",
    ),
    n_tasks: int = typer.Option(
        1, "--n-tasks", help="Number of parallel tasks to run."
    ),
    ts_mdoc_path: Path = typer.Option(
        None, help="Directory containing the tilt series mdoc file."
    ),
    ts_tomostar_path: Path = typer.Option(
        None, help="Directory containing the tomostar file."
    ),
    ts_tilt_axis: str = typer.Option(..., help="Tilt axis value."),
    ts_bin: str = typer.Option("1", help="Bin value for tilt series."),
    ts_patch_size: str = typer.Option(
        ..., help="Size of patches for patch tracking (Å)."
    ),
    ts_intensity_threshold: Optional[float] = typer.Option(
        0.75,
        "--ts-intensity-threshold",
        help=(
            "Remove tilts whose normalised intensity (relative to the lowest-dose view) "
            "drops below this value. Requires a tomostar file; if none is provided the "
            "pipeline falls back to pixel-level dark-tilt detection. "
            "Pass None to skip intensity filtering entirely."
        ),
    ),
    min_fov: float = typer.Option(0.7, help="Minimum required field of view."),
    max_attempts: int = typer.Option(3, help="Max attempts for alignment improvement."),
    is_warp_proj: bool = typer.Option(
        True,
        "--is-warp-proj/--no-warp-proj",
        help="Indicates if it is a Warp project.",
    ),
    reconstruct: bool = typer.Option(
        False, "--reconstruct", help="Reconstruct tomogram after alignment."
    ),
    ts_xml_path: Optional[Path] = typer.Option(
        None, help="Path to Warp processing results for XML update."
    ),
    tomo_bin: Optional[str] = typer.Option(
        None, help="Binned tomogram size for reconstruction."
    ),
    backup_xml: bool = typer.Option(
        True,
        "--backup-xml/--no-backup-xml",
        help="Create a backup of the XML file before updating.",
    ),
    report_file: Path = typer.Option(
        "alignment_report.txt",
        "--report-file",
        help="Path for the alignment report generated after processing completes.",
    ),
):
    """
    Perform patch-based tilt series tracking, update Warp XML, and reconstruct tomograms.
    """
    if ts_basenames:
        basenames_to_process = [name.strip() for name in ts_basenames.split(",")]
    else:
        basenames_to_process = [d.name for d in ts_data_folder.iterdir() if d.is_dir()]

    print(f"Found {len(basenames_to_process)} tilt series to process.")

    n_cpus = os.cpu_count()
    num_workers = min(n_cpus, n_tasks)
    cluster = LocalCluster(n_workers=num_workers, dashboard_address=None)
    client = Client(cluster)

    print(f"Starting Dask cluster with {num_workers} CPU workers.")

    futures_map = {}
    all_futures = []

    for basename in basenames_to_process:
        setup_future = client.submit(
            task_setup,
            basename,
            ts_data_folder,
            ts_mdoc_path,
            ts_tomostar_path,
            ts_xml_path,
            ts_tilt_axis,
            ts_bin,
            ts_patch_size,
            pure=False,
        )
        futures_map[setup_future.key] = f"{basename}: Setup"

        preprocessing_future = client.submit(
            task_preprocessing, setup_future, ts_intensity_threshold, pure=False
        )
        futures_map[preprocessing_future.key] = f"{basename}: Preprocessing"

        coarse_align_future = client.submit(
            task_coarse_alignment,
            preprocessing_future,
            min_fov,
            pure=False,
        )
        futures_map[coarse_align_future.key] = f"{basename}: Coarse Alignment"

        fine_align_future = client.submit(
            task_fine_alignment, coarse_align_future, max_attempts, pure=False
        )
        futures_map[fine_align_future.key] = f"{basename}: Fine Alignment"

        final_stack_future = client.submit(
            task_final_stack, fine_align_future, pure=False
        )
        futures_map[final_stack_future.key] = f"{basename}: Build Stack"

        base_futures = [
            setup_future,
            preprocessing_future,
            coarse_align_future,
            fine_align_future,
            final_stack_future,
        ]
        all_futures.extend(base_futures)

        last_task_future = final_stack_future
        if is_warp_proj:
            update_xml_future = client.submit(
                task_update_warp_xml,
                last_task_future,
                ts_xml_path,
                ts_tomostar_path,
                backup_xml,
                pure=False,
            )
            futures_map[update_xml_future.key] = f"{basename}: Updating XML file"
            all_futures.append(update_xml_future)

        if reconstruct:
            recon_future = client.submit(
                task_reconstruct_tomo, last_task_future, tomo_bin, pure=False
            )
            futures_map[recon_future.key] = f"{basename}: Reconstruct"
            all_futures.append(recon_future)

    pbars = {
        basename: tqdm(
            total=(5 + (1 if is_warp_proj else 0) + (1 if reconstruct else 0)),
            desc=basename,
            position=i,
        )
        for i, basename in enumerate(basenames_to_process)
    }

    future_key_to_basename = {
        f.key: futures_map.get(f.key).split(":")[0]
        for f in all_futures
        if f.key in futures_map
    }

    for future in as_completed(all_futures):
        basename = future_key_to_basename.get(future.key)
        if basename and basename in pbars:
            task_name = futures_map.get(future.key, "Unknown Task")
            if future.status == "finished":
                pbars[basename].set_postfix_str(task_name.split(":")[-1].strip())
            else:
                pbars[basename].set_postfix_str("Failed")
                #print(f"\nError in task {task_name}: {future.exception()}")
            pbars[basename].update(1)

    for pbar in pbars.values():
        pbar.close()

    client.close()
    cluster.close()
    utils.generate_alignment_report(
        ts_data_folder=ts_data_folder,
        basenames=basenames_to_process,
        output_file=report_file,
    )


if __name__ == "__main__":
    automateimod(align_tilts)
