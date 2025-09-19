import mrcfile
import typer
import starfile
import shutil
from typing import Optional
from tqdm import tqdm
import os

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils
import automateImod.pio as pio
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from dask.distributed import Client, LocalCluster, as_completed

automateimod = typer.Typer(
    no_args_is_help=True, pretty_exceptions_show_locals=False, add_completion=False
)


def task_setup(
    basename,
    ts_data_folder,
    ts_mdoc_path,
    ts_tomostar_path,
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
    return ts, logger


def task_preprocessing(setup_result):
    """Task for preprocessing: dark tilt detection and removal."""
    ts, logger = setup_result
    logger.info(f"Aligning {ts.basename}: Detecting dark frames")
    ts_path = ts.get_mrc_path()
    if not ts_path.is_file():
        logger.error(f"No valid tilt series found at {ts_path}. Aborting.")
        return None

    im_data = mrcfile.read(ts_path)
    original_tilt_frames = ts.tilt_frames.copy()
    original_tilt_angles = ts.tilt_angles.copy()

    dark_frame_indices = utils.detect_dark_tilts(
        ts_data=im_data, ts_tilt_angles=ts.tilt_angles, logger=logger
    )

    if len(dark_frame_indices) > 0:
        logger.info(f"Detected {len(dark_frame_indices)} dark tilts in {ts.basename}")
        logger.info(
            f"Aligning {ts.basename}: Removing dark frames and rebuilding stack"
        )
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
        logger.info("No dark frames found.")

    marker_file = ts.tilt_dir_name / "autoImod.marker"
    if not marker_file.exists():
        marker_file.touch()
        with open(marker_file, "w") as fout:
            fout.write("frame_basename,stage_angle,pos_in_tilt_stack\n")
            for idx in dark_frame_indices:
                if idx < len(ts.tilt_frames) and idx < len(ts.tilt_angles):
                    frame_name = ts.tilt_frames[idx]
                    fout.write(f"{frame_name},{ts.tilt_angles[idx]},{idx}\n")

    return ts, im_data, original_tilt_angles, original_tilt_frames, logger


def task_coarse_alignment(
    preprocessing_result,
    min_fov,
    max_shift_nm,
    max_shift_rate,
    use_statistical_analysis,
):
    """Task for coarse alignment and large shift detection."""
    if preprocessing_result is None:
        return None
    ts, im_data, original_tilt_angles, original_tilt_frames, logger = (
        preprocessing_result
    )
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
        logger.info(f"Aligning {ts.basename}: Removing bad tilts and rebuilding stack")

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
        marker_file = ts.tilt_dir_name / "autoImod.marker"
        with open(marker_file, "a") as fout:
            for idx in original_large_shift_indices:
                if idx < len(original_tilt_angles) and idx < len(original_tilt_frames):
                    frame_name = original_tilt_frames[idx]
                    fout.write(f"{frame_name},{original_tilt_angles[idx]},{idx}\n")
                else:
                    logger.warning(
                        f"Index {idx} is out of range for original tilt_angles or tilt_frames."
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
        return ts, logger

    logger.info(f"Residual error (nm): {resid_error}")

    if known_to_unknown > 10 and resid_error >= 1.5:
        logger.warning(f"Alignment for {ts.basename} is worse than expected.")
        for attempt in range(max_attempts):
            logger.info(
                f"Aligning {ts.basename}: Improving alignment (attempt {attempt + 1})"
            )
            utils.improve_bad_alignments(
                tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, logger=logger
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
        f"{str(ts.tilt_dir_name)}/newst_ali.com", capture_output=False
    )
    return ts, logger


def task_update_warp_xml(final_stack_result, ts_xml_path, ts_tomostar_path, backup_xml):
    """Task for updating Warp XML and TomoStar files."""
    if final_stack_result is None:
        return
    ts, logger = final_stack_result
    if not (ts_xml_path and ts_tomostar_path):
        logger.warning("Skipping Warp XML update because path is not provided.")
        return

    logger.info("Updating the provided XML metadata file and tomostar file")
    marker_file = ts.tilt_dir_name / "autoImod.marker"
    if not marker_file.exists():
        logger.warning(f"{marker_file} does not exist.")
        return

    md = pd.read_csv(marker_file, delimiter=",")
    bad_frames = md["frame_basename"].tolist()

    xml_file = ts_xml_path / f"{ts.basename}.xml"
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

    tomostar_file = ts_tomostar_path / f"{ts.basename}.tomostar"
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
        f"{str(tomo.tilt_dir_name)}/tilt_ali.com", capture_output=False
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
        True, "--is-warp-proj", help="Indicates if it is a Warp project."
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

    n_cpus = os.cpu_count()
    num_workers = min(n_cpus, n_tasks)
    cluster = LocalCluster(n_workers=num_workers)
    client = Client(cluster)

    print(f"Starting Dask cluster with {num_workers} CPU workers.")

    futures_map = {}
    all_futures = []

    for basename in basenames_to_process:
        # Chain tasks for each tilt series
        setup_future = client.submit(
            task_setup,
            basename,
            ts_data_folder,
            ts_mdoc_path,
            ts_tomostar_path,
            ts_tilt_axis,
            ts_bin,
            ts_patch_size,
            pure=False,
        )
        futures_map[setup_future.key] = f"{basename}: Setup"

        preprocessing_future = client.submit(
            task_preprocessing, setup_future, pure=False
        )
        futures_map[preprocessing_future.key] = f"{basename}: Preprocessing"

        coarse_align_future = client.submit(
            task_coarse_alignment,
            preprocessing_future,
            min_fov,
            max_shift_nm,
            max_shift_rate,
            use_statistical_analysis,
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
            futures_map[update_xml_future.key] = f"{basename}: Update XML"
            all_futures.append(update_xml_future)

        if reconstruct:
            recon_future = client.submit(
                task_reconstruct_tomo, last_task_future, tomo_bin, pure=False
            )
            futures_map[recon_future.key] = f"{basename}: Reconstruct"
            all_futures.append(recon_future)

    with tqdm(total=len(all_futures), desc="Processing Tilt Series") as pbar:
        for future in as_completed(all_futures):
            task_name = futures_map.get(future.key, "Unknown Task")
            if future.status == "finished":
                pbar.set_description(f"{task_name}")
            else:
                pbar.set_description(f"Failed {task_name}: {future.exception()}")
            pbar.update(1)

    client.close()
    cluster.close()


@automateimod.command()
def generate_alignment_report(
    ts_proc_dir: Path = typer.Option(
        ...,
        help="Directory containing the IMOD processing directories for each tilt series.",
    ),
    output_file: Path = typer.Option(
        "alignment_report.txt", help="Path for the output alignment report."
    ),
):
    """
    Compile alignment residuals into a report from 'align_patch.log' files.
    """
    print(f"Searching for tilt series directories in '{ts_proc_dir}'...")

    ts_dirs = sorted(list(ts_proc_dir.glob("*/")))

    if not ts_dirs:
        print(f"No tilt series directories found in '{ts_proc_dir}'.")
        return

    results = []
    for ts_dir in ts_dirs:
        if not ts_dir.is_dir():
            continue

        ts_name = ts_dir.name
        log_file = ts_dir / "align_patch.log"

        resid_err_val, sd_val, resid_err_wt_val = "N/A", "N/A", "N/A"

        if log_file.is_file():
            with open(log_file, "r") as f_in:
                for line in f_in:
                    if "Residual error mean and sd" in line:
                        parts = line.split()
                        if len(parts) >= 7:
                            resid_err_val = parts[5]
                            sd_val = parts[6]
                    if "error weighted mean" in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            resid_err_wt_val = parts[4]

        results.append((ts_name, resid_err_val, sd_val, resid_err_wt_val))

    if not results:
        print("No alignment logs found to generate a report.")
        return

    with open(output_file, "w") as f_out:
        f_out.write("ts_name\tresid_err\tsd\tresid_err_wt\n")
        for item in results:
            f_out.write(f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n")

    print(f"Generated alignment report at: {output_file}")


if __name__ == "__main__":
    automateimod()
