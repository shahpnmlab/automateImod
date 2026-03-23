import shutil
import subprocess
import mrcfile
import numpy as np
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import automateImod.calc as calc
import automateImod.pio as io


def build_autoimod_skip_list(
    ts_data_folder: Path, marker_name: str = "autoImod.marker"
):
    """
    Recursively scan the tilt-series data folder for marker files and return
    a sorted list of top-level tilt-series basenames to skip.
    """
    ts_data_folder = Path(ts_data_folder)
    if not ts_data_folder.exists():
        return []

    skip_basenames = set()
    try:
        for marker_path in ts_data_folder.rglob(marker_name):
            try:
                rel = marker_path.relative_to(ts_data_folder)
            except ValueError:
                continue
            # Expect marker inside a tilt-series directory: <basename>/autoImod.marker
            if len(rel.parts) >= 2:
                skip_basenames.add(rel.parts[0])
    except Exception:
        return []

    return sorted(skip_basenames)


def setup_logging(log_file_path: Path):
    """Configure logging to write to a file."""
    # Get a logger specific to the tilt series (using the file path as a key)
    logger = logging.getLogger(str(log_file_path))
    logger.setLevel(logging.INFO)

    # Prevent handlers from being added multiple times
    if not logger.handlers:
        # Create a file handler
        file_handler = logging.FileHandler(log_file_path, mode="w")
        file_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(file_handler)

    return logger


def detect_large_shifts_afterxcorr(
    coarse_align_prexg,
    image_size,
    logger,
    min_fov_fraction=0.7,
):
    """
    Simple FOV-based detection of problematic shifts.

    Args:
        coarse_align_prexg: Path to the prexg file.
        image_size: (dimX, dimY) in pixels.
        logger: Logger for diagnostics.
        min_fov_fraction: Minimum FOV fraction to keep a view.

    Returns:
        list[int]: Sorted indices of problematic views.
    """
    coarse_align_prexg = Path(coarse_align_prexg)
    prexg_data = np.loadtxt(coarse_align_prexg)
    if prexg_data.ndim == 1:
        prexg_data = np.atleast_2d(prexg_data)
    if prexg_data.shape[1] < 2:
        raise ValueError(
            f"{coarse_align_prexg} does not contain the expected shift columns."
        )

    # Extract shifts in pixels (last 2 columns: X, Y)
    shifts = prexg_data[:, -2:]

    # Get image dimensions
    dimX, dimY = image_size

    # Calculate field of view
    fov = dimX * dimY

    # Calculate area of image after shifts (clamped to 0 for out-of-frame shifts)
    area_after_shifts = np.maximum(
        0, (dimX - np.abs(shifts[:, 0])) * (dimY - np.abs(shifts[:, 1]))
    )

    # Calculate FOV fraction
    fraction_fov = area_after_shifts / fov

    # Find problematic views (below threshold)
    problematic_indices = np.where(fraction_fov < min_fov_fraction)[0].tolist()

    # Log results
    logger.info(f"Shift analysis for {coarse_align_prexg.name}:")
    logger.info(f"  Total views: {len(fraction_fov)}")
    logger.info(f"  FOV fraction threshold: {min_fov_fraction:.2f}")
    logger.info(f"  FOV range: {fraction_fov.min():.3f} - {fraction_fov.max():.3f}")
    logger.info(f"  Problematic views found: {len(problematic_indices)}")
    if problematic_indices:
        logger.info(f"  Indices: {problematic_indices}")
        for idx in problematic_indices:
            logger.info(f"    View {idx}: FOV fraction = {fraction_fov[idx]:.3f}")

    return problematic_indices


def remove_bad_tilts(ts: io.TiltSeries, im_data, pixel_nm, bad_idx):
    angpix = pixel_nm * 10
    original_rawtlt_angles = np.loadtxt(ts.get_rawtlt_path())

    # Ensure bad_idx are within the range of im_data
    bad_idx = [idx for idx in bad_idx if idx < len(im_data)]

    mask = np.ones(len(im_data), dtype=bool)
    mask[bad_idx] = False
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

    cleaned_ts_rawtlt = original_rawtlt_angles[mask]

    original_ts_file = ts.get_mrc_path()
    original_ts_rawtlt = ts.get_rawtlt_path()

    # Backup original TS data
    mrcfile.write(
        name=f"{original_ts_file}~", data=im_data, voxel_size=angpix, overwrite=True
    )
    np.savetxt(fname=f"{original_ts_rawtlt}~", X=original_rawtlt_angles, fmt="%0.2f")

    # Write new TS data
    mrcfile.write(
        name=f"{original_ts_file}", data=cleaned_mrc, voxel_size=angpix, overwrite=True
    )
    np.savetxt(fname=f"{original_ts_rawtlt}", X=cleaned_ts_rawtlt, fmt="%0.2f")

    # Update the TiltSeries object
    ts.tilt_angles = cleaned_ts_rawtlt
    ts.removed_indices = (
        sorted(set(ts.removed_indices + bad_idx))
        if hasattr(ts, "removed_indices")
        else sorted(bad_idx)
    )


def get_alignment_error(tilt_dir_name, logger):
    known_unknown_ratio = None
    resid_err = None
    sd = None

    try:
        with open(f"{tilt_dir_name}/align_patch.log", "r") as f_in:
            for line in f_in:
                if "Ratio of total measured values to all unknowns" in line:
                    known_unknown_ratio = float(line.split("=")[-1])
                if "Residual error mean and sd" in line:
                    a1 = line.split()
                    resid_err = float(a1[5])
                    sd = a1[6]

        if known_unknown_ratio is None or resid_err is None or sd is None:
            logger.warning("Could not find all alignment statistics in the log file.")

        return known_unknown_ratio, resid_err, sd

    except Exception as e:
        logger.error(f"Error reading alignment log: {e}")
        return None, None, None


def get_alignment_track_stats(tilt_dir_name):
    """
    Parse taCoordinates.log to retrieve the number of tracked contours and total points.

    Returns:
        tuple[int | None, int | None]: (unique contour count, total point count). Returns
        (None, None) if the log is missing or unreadable.
    """
    ta_log = Path(tilt_dir_name) / "taCoordinates.log"
    if not ta_log.exists():
        return None, None

    try:
        data = np.loadtxt(ta_log, skiprows=2)
    except Exception:
        return None, None

    if data.size == 0:
        return 0, 0

    if data.ndim == 1:
        data = np.atleast_2d(data)

    try:
        unique_contours = np.unique(data[:, 0]).size
    except Exception:
        return None, data.shape[0]

    return unique_contours, data.shape[0]


def write_ta_coords_log(tilt_dir_name, logger):
    with open(f"{str(tilt_dir_name)}/taCoordinates.log", "w") as ali_log:
        sbp_cmd = ["alignlog", "-c", f"{str(tilt_dir_name)}/align_patch.log"]
        write_taCoord_log = subprocess.run(
            sbp_cmd, stdout=ali_log, stderr=subprocess.PIPE, text=True
        )
        if write_taCoord_log.returncode != 0:
            if write_taCoord_log.stderr:
                logger.error(
                    f"alignlog failed (exit {write_taCoord_log.returncode}): "
                    f"{write_taCoord_log.stderr.strip()}"
                )
            else:
                logger.error(
                    f"alignlog failed with exit code {write_taCoord_log.returncode}."
                )
        elif write_taCoord_log.stderr:
            logger.warning(write_taCoord_log.stderr.strip())


def improve_bad_alignments(tilt_dir_name, tilt_name, logger):
    tilt_dir_name = str(tilt_dir_name)
    mod_file = tilt_dir_name + "/" + tilt_name + ".fid"
    mod2txt = tilt_dir_name + "/" + tilt_name + ".txt"
    txt4seed = tilt_dir_name + "/" + tilt_name + ".seed.txt"

    align_log_vals = np.loadtxt(tilt_dir_name + "/" + "taCoordinates.log", skiprows=2)

    contour_resid, median_residual = calc.median_residual(align_log_vals)

    goodpoints = np.where(contour_resid[:, [1]] <= median_residual)
    goodpoints = goodpoints[0] + 1

    # Convert the fiducial model file to a text file and readit in
    model2point_cmd = ["model2point", "-contour", mod_file, mod2txt]
    result = subprocess.run(model2point_cmd, capture_output=True, text=True)
    logger.info(f"model2point stdout: {result.stdout}")
    if result.stderr:
        logger.error(f"model2point stderr: {result.stderr}")

    #
    fid_text = np.loadtxt(mod2txt)

    # Collect rows for each good contour and assign a new sequential ID.
    # Doing this per-contour handles variable point counts correctly — the old
    # approach assumed every contour had the same number of points, which caused
    # np.repeat to produce an array of the wrong length and hstack to fail.
    chunks = []
    for new_id, old_id in enumerate(goodpoints, start=1):
        rows = fid_text[fid_text[:, 0] == old_id].copy()
        if rows.size == 0:
            logger.warning(f"Contour {old_id} not found in {mod2txt}; skipping.")
            continue
        rows[:, 0] = new_id
        chunks.append(rows)

    if not chunks:
        logger.error(
            "No valid contours remain after filtering; cannot write seed model."
        )
        return

    new_good_contours = np.vstack(chunks)

    np.savetxt(
        txt4seed, new_good_contours, fmt=" ".join(["%d"] + ["%0.3f"] * 2 + ["%d"])
    )

    point_to_model_cmd = [
        "point2model",
        "-open",
        "-circle",
        "6",
        "-image",
        f"{tilt_dir_name}/{tilt_name}_preali.mrc",
        txt4seed,
        mod_file,
    ]

    result = subprocess.run(point_to_model_cmd, capture_output=True, text=True)
    logger.info(f"point2model stdout: {result.stdout}")
    if result.stderr:
        logger.error(f"point2model stderr: {result.stderr}")


def detect_dark_tilts(
    ts_data, ts_tilt_angles, logger, brightness_factor=0.65, variance_factor=0.1
):
    # Identify the reference image, typically at or near 0 tilt
    tilt_series_mid_point_idx = calc.find_symmetric_tilt_reference(ts_tilt_angles)
    normalized_images = calc.normalize(ts_data)
    reference_image = normalized_images[tilt_series_mid_point_idx, :, :]

    # Calculate mean intensity and standard deviation for the reference image
    reference_mean_intensity = np.mean(reference_image)
    reference_std_deviation = np.std(reference_image)

    # Calculate dynamic thresholds
    threshold_intensity = reference_mean_intensity * brightness_factor
    threshold_variance = reference_std_deviation * variance_factor

    # Calculate mean intensities and standard deviations for all tilts
    mean_intensities = np.mean(normalized_images, axis=(1, 2))
    std_deviations = np.std(normalized_images, axis=(1, 2))

    # Identify dark tilts based on mean intensity and standard deviation
    dark_frame_indices = np.where(
        (mean_intensities < threshold_intensity) & (std_deviations < threshold_variance)
    )[0].tolist()

    return dark_frame_indices


def filter_by_intensities(ts: io.TiltSeries, threshold: float, logger) -> list:
    """
    Identify tilt indices whose normalised intensity falls below `threshold`.

    Normalisation is relative to the lowest-dose tilt (typically the 0° view).
    Returns:
        a list of integer indices into the *current* tilt stack;
    """
    doses = np.array(ts.doses)
    intensities = np.array(ts.intensities)
    tilt0_intensity = intensities[np.argmin(doses)]
    normed_and_clipped = np.clip((intensities / tilt0_intensity), 0, 1)
    below_threshold = np.where(normed_and_clipped <= threshold)[0].tolist()
    if below_threshold:
        logger.info(
            f"Detected {len(below_threshold)} tilt(s) with normalised intensity "
            f"< {threshold}: indices {below_threshold}, "
            f"angles {np.array(ts.tilt_angles)[below_threshold].tolist()}"
        )
    else:
        logger.info(f"No tilts below intensity threshold {threshold}.")
    return below_threshold


def swap_fast_slow_axes(tilt_dirname, tilt_name):
    (
        d,
        pixel_nm,
        _,
        _,
    ) = io.read_mrc(f"{tilt_dirname}/{tilt_name}.rec")
    d = np.swapaxes(d, 0, 1)
    mrcfile.write(
        f"{tilt_dirname}/{tilt_name}.rec",
        data=d,
        voxel_size=pixel_nm * 10,
        overwrite=True,
    )


def match_partial_filename(string_to_match, target_string, logger):
    if string_to_match in target_string:
        return True
    else:
        logger.error("Could not match string. Check if the file exists.")
        return False


def remove_xml_files(xml_file_path, logger):
    if xml_file_path.exists():
        if xml_file_path.exists():
            backup_file_path = xml_file_path.with_suffix(".xml.bkp")
            shutil.move(xml_file_path, backup_file_path)
        else:
            logger.warning(f"XML file {xml_file_path} not found.")


def validate_tilt_series_counts(ts, ts_xml_path, logger):
    """
    Verify that the frame counts from all available metadata sources agree with
    the MRC stack. Raises ValueError immediately if any mismatch is found.

    Checks performed:
      - rawtlt angle count vs MRC stack depth
      - mdoc frame count vs MRC stack depth (if mdoc was loaded)
      - tomostar entry count vs MRC stack depth (if tomostar was loaded)
      - XML angle count vs MRC stack depth (if XML path was provided)
    """
    mrc_path = ts.get_mrc_path()
    with mrcfile.open(mrc_path, mode="r", permissive=True) as mrc:
        mrc_n_frames = mrc.data.shape[0]

    mismatches = []

    rawtlt_count = len(ts.tilt_angles)
    if rawtlt_count != mrc_n_frames:
        mismatches.append(
            f"rawtlt has {rawtlt_count} angles but MRC stack has {mrc_n_frames} frames"
        )

    if ts.tilt_frames:
        frame_count = len(ts.tilt_frames)
        if frame_count != mrc_n_frames:
            mismatches.append(
                f"frame list (mdoc/tomostar) has {frame_count} entries "
                f"but MRC stack has {mrc_n_frames} frames"
            )

    if len(ts.doses) > 0 and len(ts.intensities) > 0:
        tomostar_count = len(ts.doses)
        if tomostar_count != mrc_n_frames:
            mismatches.append(
                f"tomostar has {tomostar_count} entries "
                f"but MRC stack has {mrc_n_frames} frames"
            )

    if ts_xml_path:
        xml_file = ts_xml_path / f"{ts.basename}.xml"
        if xml_file.exists():
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                angles_element = root.find("Angles")
                if angles_element is not None:
                    xml_count = len(angles_element.text.strip().split("\n"))
                    if xml_count != mrc_n_frames:
                        mismatches.append(
                            f"XML Angles element has {xml_count} entries "
                            f"but MRC stack has {mrc_n_frames} frames"
                        )
            except Exception as e:
                logger.warning(f"Could not validate XML frame count: {e}")

    if mismatches:
        msg = (
            f"Frame count mismatch for {ts.basename} — cannot proceed safely:\n"
            + "\n".join(f"  • {m}" for m in mismatches)
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        f"Frame count validation passed: all sources agree on {mrc_n_frames} frames."
    )


def generate_alignment_report(ts_data_folder: Path, basenames: list, output_file: Path):
    """
    Compile alignment residuals and final tilt counts into a formatted report.
    Reads align_patch.log and the final .rawtlt from each processed tilt series
    directory. Rows are sorted by residual error, low to high. N/A entries sort last.
    """
    results = []
    for basename in sorted(basenames):
        ts_dir = ts_data_folder / basename
        log_file = ts_dir / "align_patch.log"
        rawtlt_file = ts_dir / f"{basename}.rawtlt"

        resid_err_val = sd_val = resid_err_wt_val = "N/A"
        n_tilts_val = "N/A"

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

        if rawtlt_file.is_file():
            try:
                n_tilts_val = str(len(np.loadtxt(rawtlt_file)))
            except Exception:
                pass

        results.append((basename, n_tilts_val, resid_err_val, sd_val, resid_err_wt_val))

    # Sort by resid_err numerically; N/A entries go to the bottom
    def sort_key(row):
        try:
            return float(row[2])
        except (ValueError, TypeError):
            return float("inf")

    results.sort(key=sort_key)

    # Build formatted table with padded columns for readability
    headers = ("ts_name", "n_tilts", "resid_err", "sd", "resid_err_wt")
    col_widths = [
        max(len(headers[i]), max(len(row[i]) for row in results))
        for i in range(len(headers))
    ]

    def fmt_row(row):
        return "  ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))

    separator = "  ".join("-" * w for w in col_widths)

    with open(output_file, "w") as f_out:
        f_out.write(fmt_row(headers) + "\n")
        f_out.write(separator + "\n")
        for row in results:
            f_out.write(fmt_row(row) + "\n")

    print(f"Alignment report written to: {output_file}")


if __name__ == "__main__":
    ts_object = io.TiltSeries(
        path_to_ts_data="/Users/ps/data/wip/automateImod/example_data/Frames/imod/",
        path_to_mdoc_data="/Users/ps/data/wip/automateImod/example_data/Frames/mdoc/",
        basename="map-26-A4_ts_002",
        tilt_axis_ang=60,
        binval=10,
        patch_size=10,
    )
    a = detect_dark_tilts(ts_object)
    print(a)
