import shutil
import subprocess
import mrcfile
import numpy as np
import xml.etree.ElementTree as ET

import automateImod.calc as calc
import automateImod.pio as io


def detect_large_shifts_afterxcorr(coarse_align_prexg, shifts_threshold=1.15):
    prexg_data = []
    with open(coarse_align_prexg, "r") as file:
        for line in file:
            numbers = [float(num) for num in line.split()]
            prexg_data.append(numbers[-2:])
    prexg_data = np.array(prexg_data)
    px_shift_dist = np.sqrt(np.sum(np.square(prexg_data), axis=1))
    median_px_shift = np.median(px_shift_dist)
    whoisbigger = px_shift_dist / median_px_shift
    upper_bound = np.mean(whoisbigger) + (shifts_threshold * np.std(whoisbigger))
    large_shift_indices = np.where(whoisbigger > upper_bound)[0].tolist()
    return large_shift_indices
    # else:
    #      return np.array([])


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
        mask = mask[:len(original_rawtlt_angles)]
    elif len(mask) < len(original_rawtlt_angles):
        mask = np.pad(mask, (0, len(original_rawtlt_angles) - len(mask)), 'constant', constant_values=True)

    cleaned_ts_rawtlt = original_rawtlt_angles[mask]

    original_ts_file = ts.get_mrc_path()
    original_ts_rawtlt = ts.get_rawtlt_path()

    # Backup original TS data
    mrcfile.write(name=f'{original_ts_file}~', data=im_data, voxel_size=angpix, overwrite=True)
    np.savetxt(fname=f'{original_ts_rawtlt}~', X=original_rawtlt_angles, fmt='%0.2f')

    # Write new TS data
    mrcfile.write(name=f'{original_ts_file}', data=cleaned_mrc, voxel_size=angpix, overwrite=True)
    np.savetxt(fname=f'{original_ts_rawtlt}', X=cleaned_ts_rawtlt, fmt='%0.2f')

    # Update the TiltSeries object
    ts.tilt_angles = cleaned_ts_rawtlt
    ts.removed_indices = sorted(set(ts.removed_indices + bad_idx)) if hasattr(ts, 'removed_indices') else sorted(
        bad_idx)


def get_alignment_error(tilt_dir_name):
    known_unknown_ratio = None
    resid_err = None
    sd = None

    try:
        with open(f'{tilt_dir_name}/align_patch.log', 'r') as f_in:
            for line in f_in:
                if "Ratio of total measured values to all unknowns" in line:
                    known_unknown_ratio = float(line.split("=")[-1])
                if "Residual error mean and sd" in line:
                    a1 = line.split()
                    resid_err = float(a1[5])
                    sd = a1[6]

        if known_unknown_ratio is None or resid_err is None or sd is None:
            print("Warning: Could not find all alignment statistics in the log file.")

        return known_unknown_ratio, resid_err, sd

    except Exception as e:
        print(f"Error reading alignment log: {e}")
        return None, None, None


def write_ta_coords_log(tilt_dir_name):
    with open(f"{str(tilt_dir_name)}/taCoordinates.log", "w") as ali_log:
        sbp_cmd = ['alignlog', '-c', f'{str(tilt_dir_name)}/align_patch.log']
        write_taCoord_log = subprocess.run(sbp_cmd, stdout=ali_log,
                                           stderr=subprocess.PIPE,
                                           text=True)
        print(write_taCoord_log.stdout)
        print(write_taCoord_log.stderr)


def improve_bad_alignments(tilt_dir_name, tilt_name):
    tilt_dir_name = str(tilt_dir_name)
    mod_file = tilt_dir_name + "/" + tilt_name + ".fid"
    mod2txt = tilt_dir_name + "/" + tilt_name + ".txt"
    txt4seed = tilt_dir_name + "/" + tilt_name + ".seed.txt"

    align_log_vals = np.loadtxt(tilt_dir_name + "/" + "taCoordinates.log", skiprows=2)

    contour_resid, median_residual = calc.median_residual(align_log_vals)

    goodpoints = np.where(contour_resid[:, [1]] <= np.around(median_residual))
    goodpoints = goodpoints[0] + 1

    # Convert the fiducial model file to a text file and readit in
    subprocess.run(['model2point', '-contour', mod_file, mod2txt])

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

    np.savetxt(txt4seed, new_good_contours, fmt=' '.join(['%d'] + ['%0.3f'] * 2 + ['%d']))

    point_to_model_cmd = ['point2model', '-open', '-circle', '6', '-image', f"{tilt_dir_name}/{tilt_name}_preali.mrc",
                          txt4seed, mod_file]

    subprocess.run(point_to_model_cmd)


def detect_dark_tilts(ts_data, ts_tilt_angles, brightness_factor=0.65, variance_factor=0.1):
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
    dark_frame_indices = np.where((mean_intensities < threshold_intensity) & (std_deviations < threshold_variance))[0].tolist()

    return dark_frame_indices

def swap_fast_slow_axes(tilt_dirname, tilt_name):
    d, pixel_nm, _, _, = io.read_mrc(f'{tilt_dirname}/{tilt_name}.rec')
    d = np.swapaxes(d, 0, 1)
    mrcfile.write(f'{tilt_dirname}/{tilt_name}.rec', data=d, voxel_size=pixel_nm * 10, overwrite=True)


def match_partial_filename(string_to_match, target_string):
    if string_to_match in target_string:
        return True
    else:
        print("Could not match string. Check if the file exists.")
        return False


# def update_xml_files(xml_file_path):
#     if xml_file_path.exists():
#         tree = ET.parse(xml_file_path)
#         root = tree.getroot()
#         if 'UnselectFilter' in root.attrib:
#             root.set('UnselectManual', str(True))
#             tree.write(xml_file_path)
#     else:
#         print(f"XML file {xml_file_path} not found.")

def remove_xml_files(xml_file_path):
    if xml_file_path.exists():
        if xml_file_path.exists():
            backup_file_path = xml_file_path.with_suffix('.xml.bkp')
            shutil.move(xml_file_path, backup_file_path)
        else:
            print(f"XML file {xml_file_path} not found.")


if __name__ == '__main__':
    ts_object = io.TiltSeries(path_to_ts_data="/Users/ps/data/wip/automateImod/example_data/Frames/imod/",
                              path_to_mdoc_data="/Users/ps/data/wip/automateImod/example_data/Frames/mdoc/",
                              basename="map-26-A4_ts_002",
                              tilt_axis_ang=60, binval=10, patch_size=10)
    a = detect_dark_tilts(ts_object)
    print(a)
