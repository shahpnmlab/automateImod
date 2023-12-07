from typing import Union

import mrcfile
import mdocfile
import subprocess
import numpy as np
from pathlib import Path, PureWindowsPath

import calc
import ps_io


def detect_dark_tilts(mrcin, mdocf, sum_tolerance=0.01):
    mdoc_md = mdocfile.read(mdocf)
    im_data, _, _, _ = ps_io.read_mrc(mrcin)

    normalised_sums = calc.normalised_sums(im_data=im_data)
    sum_close_to_zero_indices = np.where(np.abs(normalised_sums) <= sum_tolerance)[0]
    valid_dark_tilts_idx = [str(idx) for idx in sum_close_to_zero_indices]

    dark_tilts_data = mdoc_md.sort_values(by=["TiltAngle"]).iloc[sum_close_to_zero_indices]
    dark_tilts_angle = dark_tilts_data["TiltAngle"].astype(str)
    dark_tilts_frames = dark_tilts_data["SubFramePath"].apply(lambda p: PureWindowsPath(p).name)

    return valid_dark_tilts_idx, dark_tilts_angle, dark_tilts_frames


def detect_large_shifts_afterxcorr(coarse_align_prexf, sdev=4):
    prexf_data = []
    with open(coarse_align_prexf, "r") as file:
        for line in file:
            numbers = [float(num) for num in line.split()]
            prexf_data.append(numbers[-2:])
    prexf_data = np.array(prexf_data)
    summed_shifts = np.abs(np.sum(prexf_data, axis=1))
    # Calculate mean and standard deviation
    mean_shift = np.mean(summed_shifts)
    std_shift = np.std(summed_shifts)
    threshold = mean_shift + sdev * std_shift
    large_shift_indices = np.where(summed_shifts > threshold)[0]
    return large_shift_indices


def remove_tilts_with_large_shifts(tilt_dir_name, tilt_name, tilt_extension, tilt_rawtlt):
    # Setup path file names
    original_ts_file = tilt_dir_name() / f'{tilt_name}.{tilt_extension}'
    original_ts_rawtlt = tilt_dir_name() / f'{tilt_rawtlt}'
    coarse_align_prexf = tilt_dir_name() / f'{tilt_name}.prexf'
    marker_file = tilt_dir_name() / ".removed_tilts_with_large_shifts"

    if not marker_file.exists():
        # Grab data
        original_mrc_data, pixel_nm, _, _ = ps_io.read_mrc(original_ts_file)
        angpix = pixel_nm * 10

        # Detect tilts with large shifts
        bad_tilt_indices = detect_large_shifts_afterxcorr(coarse_align_prexf=coarse_align_prexf)
        original_rawtlt_angles = np.loadtxt(original_ts_rawtlt)
        mask = np.ones(len(original_mrc_data), dtype=bool)
        mask[bad_tilt_indices] = False
        bad_tilts = original_rawtlt_angles[bad_tilt_indices]
        print("Saving a backup of the original TS and rawtlt files.")
        print("Removing tilts with large shifts...")
        print(f"Tilt series path: {original_ts_file}")
        print(f"Views with large shifts: {', '.join(map(lambda tilt: f'{tilt}˚', bad_tilts))}")
        cleaned_mrc = original_mrc_data[mask]
        cleaned_ts_rawtlt = original_rawtlt_angles[mask]

        # Backup original TS data
        mrcfile.write(name=f'{original_ts_file}~', data=original_mrc_data, voxel_size=angpix, overwrite=True)
        np.savetxt(fname=f'{original_ts_rawtlt}~', X=original_rawtlt_angles, fmt='%0.2f')

        # Write new TS data
        mrcfile.write(name=f'{original_ts_file}', data=cleaned_mrc, voxel_size=angpix, overwrite=True)
        np.savetxt(fname=f'{original_ts_rawtlt}', X=cleaned_ts_rawtlt, fmt='%0.2f')

        # Mark this folder as done
        Path(f"{tilt_dir_name()}/.removed_tilts_with_large_shifts").touch()


def get_alignment_error(tilt_dir_name):
    with open(f'{tilt_dir_name}/align_patch.log', 'r') as f_in:
        for line in f_in:
            if "Ratio of total measured values to all unknowns" in line:
                known_unknown_ratio = float(line.split("=")[-1])
            if "Residual error mean and sd" in line:
                a1 = line.split()
                resid_err = float(a1[5])
                sd = a1[6]
            # if "error weighted mean" in line:
            #     a2 = line.split()
            #     resid_err_wt.append(a2[4])
    return known_unknown_ratio, resid_err, sd


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

    subprocess.run(['point2model', '-open', '-circle', '6','-image', tilt_dir_name + "/" + tilt_name + "_preali.mrc", txt4seed, mod_file])


def swap_fast_slow_axes(tilt_dirname, tilt_name):
    d, pixel_nm, _,_, = ps_io.read_mrc(f'{tilt_dirname}/{tilt_name}.rec')
    d = np.swapaxes(d,0,1)
    mrcfile.write(f'{tilt_dirname}/{tilt_name}.rec', data=d, voxel_size=pixel_nm*10, overwrite=True)
