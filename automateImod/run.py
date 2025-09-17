import mrcfile
import typer
import starfile
import shutil
from typing import List, Optional
import dask
from tqdm import tqdm
import logging

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils
import automateImod.pio as pio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path

automateImod = typer.Typer(
    no_args_is_help=True, pretty_exceptions_show_locals=False, add_completion=False
)


def align_tilt_series(
    ts: pio.TiltSeries,
    min_fov: float,
    max_shift_nm: float,
    max_shift_rate: float,
    use_statistical_analysis: bool,
    max_attempts: int,
    logger: logging.Logger,
):
    """Helper function to perform the tilt series alignment."""
    ts_path = ts.get_mrc_path()
    marker_file = ts.tilt_dir_name / "autoImod.marker"

    if not ts_path.is_file():
        logger.error(f"No valid tilt series found at {ts_path}. Aborting.")
        return

    im_data = mrcfile.read(ts_path)

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
        pixel_nm=ts.pixel_size / 10,
        binval=ts.binval,
        tilt_axis_ang=ts.tilt_axis_ang,
        dimX=ts.dimX,
        dimY=ts.dimY,
    )
    coms.make_ali_stack_com(
        tilt_dir_name=ts.tilt_dir_name,
        tilt_name=ts.basename,
        tilt_extension=ts.extension,
        binval=ts.binval,
        dimX=ts.dimX,
        dimY=ts.dimY,
    )

    original_tilt_frames = ts.tilt_frames.copy()
    original_tilt_angles = ts.tilt_angles.copy()

    if not marker_file.exists():
        marker_file.touch()
        logger.info(f"Marker file not detected in {marker_file.parent}\nProcessing...")
        logger.info("Looking for dark tilts...")

        dark_frame_indices = utils.detect_dark_tilts(
            ts_data=im_data, ts_tilt_angles=ts.tilt_angles, logger=logger
        )

        if len(dark_frame_indices) > 0:
            logger.info(
                f"Detected {len(dark_frame_indices)} dark tilts in {ts.basename}"
            )
            logger.info("Removing dark tilts...")
            utils.remove_bad_tilts(
                ts=ts,
                im_data=im_data,
                pixel_nm=ts.pixel_size / 10,
                bad_idx=dark_frame_indices,
            )
            ts.removed_indices.extend(dark_frame_indices)
            del im_data
            im_data, _, _, _ = pio.read_mrc(ts_path)
        else:
            logger.info("No dark frames found. Proceeding with coarse alignments...")

        with open(marker_file, "w") as fout:
            fout.write("frame_basename,stage_angle,pos_in_tilt_stack\n")
            for idx in dark_frame_indices:
                if idx < len(ts.tilt_frames) and idx < len(ts.tilt_angles):
                    frame_name = ts.tilt_frames[idx]
                    fout.write(f"{frame_name},{ts.tilt_angles[idx]},{idx}\n")

        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/xcorr_coarse.com", capture_output=False
        )
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/newst_coarse.com", capture_output=False
        )

        large_shift_indices = utils.detect_large_shifts_afterxcorr(
            coarse_align_prexg=f"{ts.tilt_dir_name}/{ts.basename}.prexg",
            pixel_size_nm=ts.pixel_size / 10,
            image_size=(ts.dimX, ts.dimY),
            logger=logger,
            min_fov_fraction=min_fov,
            max_shift_nm=max_shift_nm,
            max_shift_rate=max_shift_rate,
            use_statistical_analysis=use_statistical_analysis,
        )

        if len(large_shift_indices) > 0:
            logger.info(
                f"Detected {len(large_shift_indices)} badly tracking tilts in {ts.basename}"
            )
            logger.info("Removing badly tracked tilts...")

            original_large_shift_indices = [
                np.where(np.array(original_tilt_angles) == ts.tilt_angles[idx])[0][0]
                for idx in large_shift_indices
            ]

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

            with open(marker_file, "a") as fout:
                for idx in original_large_shift_indices:
                    if idx < len(original_tilt_angles) and idx < len(
                        original_tilt_frames
                    ):
                        frame_name = original_tilt_frames[idx]
                        fout.write(f"{frame_name},{original_tilt_angles[idx]},{idx}\n")
                    else:
                        logger.warning(
                            f"Index {idx} is out of range for original tilt_angles or tilt_frames."
                        )

    ts.removed_indices = sorted(set(ts.removed_indices))

    logger.info(f"Performing patch-based alignment on {ts.basename}")
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/xcorr_patch.com", capture_output=False
    )
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
    )
    utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name, logger=logger)

    known_to_unknown, resid_error, sd = utils.get_alignment_error(
        ts.tilt_dir_name, logger
    )

    if resid_error is None:
        logger.warning("Alignment failed, could not retrieve statistics.")
        return

    logger.info(f"Residual error (nm): {resid_error}")

    if known_to_unknown > 10 and resid_error >= 1.5:
        logger.warning(f"Alignment for {ts.basename} is worse than expected.")
        for attempt in range(max_attempts):
            logger.info(f"Improving alignment, attempt {attempt + 1}")
            utils.improve_bad_alignments(
                tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename
            )
            coms.execute_com_file(
                f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
            )
            utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name, logger=logger)
            _, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name, logger)
            if resid_error is not None and resid_error < 1.5:
                logger.info(f"Alignment improved. Residual error: {resid_error} nm")
                break
        else:
            logger.warning("Max realignment attempts reached.")

    logger.info(
        f"Final alignment accuracy: Residual error (nm): {resid_error} (SD: {sd})"
    )
    logger.info(f"Building aligned stack for {ts.basename}...")
    coms.execute_com_file(
        f"{str(ts.tilt_dir_name)}/newst_ali.com", capture_output=False
    )


def update_warp_xml_and_tomostar(
    ts_basename: str,
    ts_xml_path: Path,
    ts_tomostar_path: Path,
    ts_log_path: Path,
    backup_xml: bool,
    logger: logging.Logger,
):
    """Helper function to update Warp XML and TomoStar files."""
    logger.info("Updating the provided XML metadata file and tomostar file")

    marker_file = ts_log_path / f"{ts_basename}/autoImod.marker"
    if not marker_file.exists():
        logger.warning(f"{marker_file} does not exist.")
        return

    md = pd.read_csv(marker_file, delimiter=",")
    bad_frames = md["frame_basename"].tolist()

    xml_file = ts_xml_path / f"{ts_basename}.xml"
    if not xml_file.exists():
        logger.error(f"XML file {xml_file} does not exist.")
        return

    if backup_xml:
        backup_file_path = xml_file.with_suffix(xml_file.suffix + ".bak")
        try:
            shutil.copy2(xml_file, backup_file_path)
            logger.info(f"Backup of {xml_file} created at {backup_file_path}")
        except Exception as e:
            logger.error(f"Error creating backup for {xml_file}: {e}")

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
            movie_paths = root.find("MoviePath").text.strip().split("\n")
            for line, movie in zip(lines, movie_paths):
                frame_name = Path(movie).stem
                if frame_name not in bad_frames:
                    updated_lines.append(line)
            element.text = "\n" + "\n".join(updated_lines) + "\n"
        else:
            logger.warning(f"{element_name} element not found in the XML file.")

    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    logger.info(f"Updated XML file: {xml_file}")

    tomostar_file = ts_tomostar_path / f"{ts_basename}.tomostar"
    if not tomostar_file.exists():
        logger.error(f"Tomostar file {tomostar_file} does not exist.")
        return

    tomostar_data = starfile.read(tomostar_file)
    good_mask = (
        ~tomostar_data["wrpMovieName"].apply(lambda x: Path(x).name).isin(bad_frames)
    )
    updated_tomostar_data = tomostar_data[good_mask]
    starfile.write(updated_tomostar_data, tomostar_file, overwrite=True)
    logger.info(f"Updated tomostar file: {tomostar_file}")


def reconstruct_tomo(
    ts_data_path: Path,
    ts_basename: str,
    tomo_bin: str,
    ts_extension: str = "mrc",
):
    """Helper function to reconstruct a tomogram."""
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

    coms.execute_com_file(f"{str(tomo.tilt_dir_name)}/newst_ali.com")
    coms.execute_com_file(f"{str(tomo.tilt_dir_name)}/tilt_ali.com")
    utils.swap_fast_slow_axes(tomo.tilt_dir_name, tomo.basename)


def process_tilt_series(
    basename: str,
    ts_data_folder: Path,
    ts_mdoc_path: Path,
    ts_tomostar_path: Path,
    ts_tilt_axis: str,
    ts_bin: str,
    ts_patch_size: str,
    min_fov: float,
    max_attempts: int,
    max_shift_nm: float,
    max_shift_rate: float,
    use_statistical_analysis: bool,
    is_warp_proj: bool,
    reconstruct: bool,
    ts_xml_path: Optional[Path],
    tomo_bin: Optional[str],
    backup_xml: bool,
):
    """
    Processes a single tilt series, including alignment, XML update, and reconstruction.
    """
    ts_data_path = ts_data_folder
    ts_basename = basename

    log_path = ts_data_path / ts_basename / f"{ts_basename}.log"
    logger = utils.setup_logging(log_path)

    logger.info(f"Starting processing for tilt series: {ts_basename}")

    try:
        ts = pio.TiltSeries(
            path_to_ts_data=ts_data_path,
            path_to_mdoc_data=ts_mdoc_path,
            path_to_tomostar=ts_tomostar_path,
            basename=ts_basename,
            tilt_axis_ang=ts_tilt_axis,
            binval=ts_bin,
            patch_size_ang=ts_patch_size,
            logger=logger,
        )

        align_tilt_series(
            ts=ts,
            min_fov=min_fov,
            max_shift_nm=max_shift_nm,
            max_shift_rate=max_shift_rate,
            use_statistical_analysis=use_statistical_analysis,
            max_attempts=max_attempts,
            logger=logger,
        )

        if is_warp_proj:
            if ts_xml_path and ts_tomostar_path:
                update_warp_xml_and_tomostar(
                    ts_basename=ts_basename,
                    ts_xml_path=ts_xml_path,
                    ts_tomostar_path=ts_tomostar_path,
                    ts_log_path=ts.tilt_dir_name.parent,
                    backup_xml=backup_xml,
                    logger=logger,
                )
            else:
                logger.warning("Skipping Warp XML update because path is not provided.")

        if reconstruct:
            if tomo_bin:
                reconstruct_tomo(
                    ts_data_path=ts.tilt_dir_name.parent,
                    ts_basename=ts_basename,
                    tomo_bin=tomo_bin,
                )
            else:
                logger.warning(
                    "Skipping reconstruction because tomo_bin is not provided."
                )

    except Exception as e:
        logger.error(f"An error occurred during processing of {ts_basename}: {e}")


@automateImod.command(no_args_is_help=True)
def run(
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
    min_fov: float = typer.Option(0.7, help="Minimum required field of view."),
    max_attempts: int = typer.Option(3, help="Max attempts for alignment improvement."),
    max_shift_nm: float = typer.Option(
        None, help="Maximum acceptable absolute shift in nm."
    ),
    max_shift_rate: float = typer.Option(
        50.0, help="Maximum acceptable shift rate in nm/degree."
    ),
    use_statistical_analysis: bool = typer.Option(
        True, help="Use statistical outlier detection for shifts."
    ),
    is_warp_proj: bool = typer.Option(
        False, "--is-warp-proj", help="Indicates if it is a Warp project."
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
):
    """
    Perform patch-based tilt series tracking, update Warp XML, and reconstruct tomograms.
    """
    if ts_basenames:
        basenames_to_process = [name.strip() for name in ts_basenames.split(",")]
    else:
        basenames_to_process = [d.name for d in ts_data_folder.iterdir() if d.is_dir()]

    print(f"Found {len(basenames_to_process)} tilt series to process.")

    delayed_tasks = [
        dask.delayed(process_tilt_series)(
            basename=basename,
            ts_data_folder=ts_data_folder,
            ts_mdoc_path=ts_mdoc_path,
            ts_tomostar_path=ts_tomostar_path,
            ts_tilt_axis=ts_tilt_axis,
            ts_bin=ts_bin,
            ts_patch_size=ts_patch_size,
            min_fov=min_fov,
            max_attempts=max_attempts,
            max_shift_nm=max_shift_nm,
            max_shift_rate=max_shift_rate,
            use_statistical_analysis=use_statistical_analysis,
            is_warp_proj=is_warp_proj,
            reconstruct=reconstruct,
            ts_xml_path=ts_xml_path,
            tomo_bin=tomo_bin,
            backup_xml=backup_xml,
        )
        for basename in basenames_to_process
    ]

    with tqdm(total=len(delayed_tasks), desc="Processing Tilt Series") as pbar:
        results = dask.compute(*delayed_tasks, num_workers=n_tasks)
        for _ in results:
            pbar.update(1)


if __name__ == "__main__":
    automateImod()
