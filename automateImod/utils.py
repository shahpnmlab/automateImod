import shutil
import subprocess
import mrcfile
import numpy as np
from pathlib import Path
import logging

import automateImod.calc as calc
import automateImod.pio as io


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
    pixel_size_nm,
    image_size,
    logger,
    tilt_angles,
    min_fov_fraction=0.8,
    max_shift_nm=None,
    max_shift_rate=50.0,
    outlier_threshold=3.0,
    min_acceptable_overlap=0.5,
    use_statistical_analysis=True,
):
    """
    Robust detection of problematic shifts using multiple criteria.

    Args:
        coarse_align_prexg: Path to the prexg file.
        pixel_size_nm: Pixel size in nanometres.
        image_size: (width, height) in pixels for the aligned stack.
        logger: Logger used for reporting diagnostics.
        tilt_angles: Sequence/array of tilt angles corresponding to the views in the stack.
        min_fov_fraction: Minimum desired overlap fraction before flagging a view.
        max_shift_nm: Maximum acceptable absolute shift in nm. If None, defaults to 20% of the smaller field dimension.
        max_shift_rate: Maximum acceptable shift rate in nm/degree.
        outlier_threshold: Threshold in MAD units for statistical overlap outlier detection.
        min_acceptable_overlap: Hard lower bound on allowable overlap.
        use_statistical_analysis: Enable robust statistical screening of overlap values.

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

    shifts_pixels = prexg_data[:, -2:]
    shifts_nm = shifts_pixels * float(pixel_size_nm)
    shift_magnitudes = np.linalg.norm(shifts_nm, axis=1)

    width_nm = float(image_size[0]) * float(pixel_size_nm)
    height_nm = float(image_size[1]) * float(pixel_size_nm)

    if max_shift_nm is None:
        max_shift_nm = 0.2 * min(width_nm, height_nm)

    x_overlaps = 1 - np.abs(shifts_nm[:, 0]) / width_nm
    y_overlaps = 1 - np.abs(shifts_nm[:, 1]) / height_nm
    min_overlaps = np.minimum(x_overlaps, y_overlaps)

    tilt_angles = np.asarray(tilt_angles, dtype=float)
    if tilt_angles.shape[0] != shifts_nm.shape[0]:
        raise ValueError(
            "Number of tilt angles does not match number of views in the prexg file."
        )

    problematic_views = set()
    breakdown = {}

    abs_bad_indices = np.where(min_overlaps < min_acceptable_overlap)[0]
    if abs_bad_indices.size:
        problematic_views.update(abs_bad_indices.tolist())
    breakdown["min_overlap"] = abs_bad_indices

    max_shift_indices = np.where(shift_magnitudes > max_shift_nm)[0]
    if max_shift_indices.size:
        problematic_views.update(max_shift_indices.tolist())
    breakdown["max_shift"] = max_shift_indices

    rate_exceed_pairs = np.array([], dtype=int)
    shift_rates = np.array([], dtype=float)
    delta_angles = np.array([], dtype=float)

    if shifts_nm.shape[0] > 1:
        shift_diffs = np.diff(shifts_nm, axis=0)
        shift_diff_magnitudes = np.linalg.norm(shift_diffs, axis=1)

        delta_angles = np.abs(np.diff(tilt_angles))
        with np.errstate(divide="ignore", invalid="ignore"):
            shift_rates = np.divide(
                shift_diff_magnitudes,
                delta_angles,
                out=np.full_like(shift_diff_magnitudes, np.nan),
                where=delta_angles > 0,
            )

        if max_shift_rate is not None:
            jump_thresholds_nm = max_shift_rate * delta_angles
            rate_exceed_pairs = np.where(
                (delta_angles > 0) & (shift_diff_magnitudes > jump_thresholds_nm)
            )[0]
            for idx in rate_exceed_pairs:
                problematic_views.update((idx, idx + 1))
    breakdown["shift_rate_pairs"] = rate_exceed_pairs

    statistical_outlier_indices = np.array([], dtype=int)
    if use_statistical_analysis:
        current_good_indices = np.array(
            sorted(set(range(len(min_overlaps))) - problematic_views)
        )
        remaining_overlaps = min_overlaps[current_good_indices]

        if remaining_overlaps.size > 5:
            median_overlap = np.median(remaining_overlaps)
            mad = np.median(np.abs(remaining_overlaps - median_overlap))
            if mad > 0:
                threshold = median_overlap - outlier_threshold * mad * 1.4826
                mask = (min_overlaps < threshold) & (min_overlaps < min_fov_fraction)
                statistical_outlier_indices = np.where(mask)[0]
                problematic_views.update(statistical_outlier_indices.tolist())
    breakdown["overlap_outliers"] = statistical_outlier_indices

    consistency_indices = []
    if shifts_nm.shape[0] > 2:
        for i in range(1, shifts_nm.shape[0] - 1):
            if i in problematic_views:
                continue

            prev_shift = shifts_nm[i - 1]
            curr_shift = shifts_nm[i]
            next_shift = shifts_nm[i + 1]

            prev_angle = tilt_angles[i - 1]
            curr_angle = tilt_angles[i]
            next_angle = tilt_angles[i + 1]
            span = next_angle - prev_angle
            if span == 0:
                continue

            weight = (curr_angle - prev_angle) / span
            expected_shift = prev_shift + weight * (next_shift - prev_shift)
            deviation = np.linalg.norm(curr_shift - expected_shift)

            left_span = abs(curr_angle - prev_angle)
            right_span = abs(next_angle - curr_angle)
            local_span = max(min(left_span, right_span), 1e-6)
            allowed_deviation = (
                max_shift_rate * local_span
                if max_shift_rate is not None
                else max_shift_nm
            )

            if deviation > allowed_deviation and min_overlaps[i] < min_fov_fraction:
                problematic_views.add(i)
                consistency_indices.append(i)
    breakdown["consistency"] = np.array(consistency_indices, dtype=int)

    sorted_problematic = sorted(problematic_views)

    logger.info(f"Shift analysis for {coarse_align_prexg.name}:")
    logger.info(f"  Total views: {shifts_nm.shape[0]}")
    logger.info(f"  Maximum shift threshold: {max_shift_nm:.1f} nm")
    if shifts_nm.shape[0] > 1 and max_shift_rate is not None:
        logger.info(f"  Shift-rate threshold: {max_shift_rate:.1f} nm/deg")
        finite_rates = shift_rates[np.isfinite(shift_rates)]
        if finite_rates.size:
            logger.info(
                f"  Observed shift rates (nm/deg): min {np.min(finite_rates):.1f}, "
                f"median {np.median(finite_rates):.1f}, max {np.max(finite_rates):.1f}"
            )
    logger.info(f"  Problematic views found: {len(sorted_problematic)}")
    if sorted_problematic:
        logger.info(f"  Indices: {sorted_problematic}")
        logger.info("  Criteria breakdown:")
        if breakdown["min_overlap"].size:
            logger.info(
                f"    - Absolute minimum overlap violations "
                f"({min_acceptable_overlap*100:.0f}%): {breakdown['min_overlap'].size}"
            )
        if breakdown["max_shift"].size:
            logger.info(
                f"    - Maximum shift exceeded ({max_shift_nm:.1f} nm): "
                f"{breakdown['max_shift'].size}"
            )
        if breakdown["shift_rate_pairs"].size:
            logger.info(
                f"    - Shift-rate violations: {breakdown['shift_rate_pairs'].size} transitions"
            )
        if use_statistical_analysis and breakdown["overlap_outliers"].size:
            logger.info(
                f"    - Statistical overlap outliers: {breakdown['overlap_outliers'].size}"
            )
        if breakdown["consistency"].size:
            logger.info(
                f"    - Local consistency deviations: {breakdown['consistency'].size}"
            )

    return sorted_problematic


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

    goodpoints = np.where(contour_resid[:, [1]] <= np.around(median_residual))
    goodpoints = goodpoints[0] + 1

    # Convert the fiducial model file to a text file and readit in
    model2point_cmd = ["model2point", "-contour", mod_file, mod2txt]
    result = subprocess.run(model2point_cmd, capture_output=True, text=True)
    logger.info(f"model2point stdout: {result.stdout}")
    if result.stderr:
        logger.error(f"model2point stderr: {result.stderr}")

    fid_text = np.loadtxt(mod2txt)
    new_good_contours = np.empty((0, 4))

    for i in range(goodpoints.shape[0]):
        a = fid_text[fid_text[:, 0] == goodpoints[i]]
        new_good_contours = np.append(new_good_contours, a, axis=0)

    number_of_new_contours = np.unique(new_good_contours[:, 0]).shape[0]
    number_of_new_contours = np.arange(number_of_new_contours) + 1

    repeats = np.unique(new_good_contours[:, -1]).shape[0]
    new_contour_id = np.repeat(number_of_new_contours, repeats)
    new_contour_id = new_contour_id.reshape(new_contour_id.shape[0], 1)
    new_good_contours = np.hstack((new_contour_id, new_good_contours[:, 1:]))

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
