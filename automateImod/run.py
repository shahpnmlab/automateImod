import typer
import subprocess
import pandas as pd
from pathlib import Path

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils
import automateImod.pio as pio

# TODO: 4. Surely warp can be tricked into accepting decimated stack results...
# TODO: 4. bazooka - remove offending entries from the tomostar file AND rename the xml file for the frame.

automateImod = typer.Typer()


@automateImod.command(no_args_is_help=True)
def align_tilts(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                ts_mdoc_path: Path = typer.Option(None, help="directory containing the tilt series mdoc file"),
                ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_1"),
                ts_tilt_axis: str = typer.Option(..., help="tilt axis value"),
                ts_bin: str = typer.Option(..., help="bin value to reduce the tilt series size by."),
                ts_patch_size: str = typer.Option(..., help="Size of patches to perform patch_tracking")):
    """
    Perform patch-based tilt series tracking using IMOD routines
    """
    # Initialise a TS object with user inputs
    ts = pio.TiltSeries(path_to_ts_data=ts_data_path, path_to_mdoc_data=ts_mdoc_path, basename=ts_basename,
                        tilt_axis_ang=ts_tilt_axis, binval=ts_bin, patch_size=ts_patch_size)
    ts_path = ts.get_mrc_path()
    marker_file = ts.tilt_dir_name / "autoImod.marker"

    if ts_path.is_file():
        im_data, pixel_nm, dimX, dimY = pio.read_mrc(ts_path)

        coms.write_xcorr_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, tilt_extension=ts.extension,
                             tilt_axis_ang=ts.tilt_axis_ang)
        coms.make_xcorr_stack_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, tilt_extension=ts.extension,
                                  binval=ts.binval)
        coms.make_patch_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, binval=ts.binval,
                            tilt_axis_ang=ts.tilt_axis_ang, patch=ts.patch_size)
        coms.track_patch_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, pixel_nm=pixel_nm, binval=ts.binval,
                             tilt_axis_ang=ts.tilt_axis_ang, dimX=dimX, dimY=dimY)

        if not marker_file.exists():
            marker_file.touch()
            print(f"Marker file not detected in {marker_file.parent}\nProcessing...")
            print("Looking for dark tilts...")

            dark_frame_indices = utils.detect_dark_tilts(ts_data=im_data, ts_tilt_angles=ts.read_rawtlt_file())

            if len(dark_frame_indices) > 0:
                print(f'Detected {len(dark_frame_indices)} dark tilts in {ts.basename}')
                print(f'Removing dark tilts...')
                utils.remove_bad_tilts(ts=ts, im_data=im_data, pixel_nm=pixel_nm, bad_idx=dark_frame_indices)
                ts.remove_frames(dark_frame_indices)
                del im_data
                im_data, pixel_nm, dimX, dimY = pio.read_mrc(ts_path)
            else:
                print("No dark frames found in the current tilt series. Proceeding with coarse alignments...")

            coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_coarse.com', capture_output=False)
            coms.execute_com_file(f'{str(ts.tilt_dir_name)}/newst_coarse.com', capture_output=False)

            with open(marker_file, "w") as fout:
                fout.write("frame_basename,stage_angle,pos_in_tilt_stack\n")
                for idx, value in enumerate(dark_frame_indices):
                    fout.write(f"{ts.tilt_frames[value]},{ts.tilt_angles[value]},{dark_frame_indices[idx]}\n")

            large_shift_indices = utils.detect_large_shifts_afterxcorr(f'{ts.tilt_dir_name}/{ts.basename}.prexg')

            if len(large_shift_indices) > 0:
                print(f'Detected {len(large_shift_indices)} badly tracking tilts in {ts.basename}')
                print(f'Removing badly tracked tilts...')
                utils.remove_bad_tilts(ts=ts, im_data=im_data, pixel_nm=pixel_nm, bad_idx=large_shift_indices)
                print(f"Redoing coarse alignment with decimated {ts.basename} stack")

                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_coarse.com', capture_output=False)
                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/newst_coarse.com', capture_output=False)

                with open(marker_file, "a") as fout:
                    for idx, value in enumerate(large_shift_indices):
                        fout.write(f"{ts.tilt_frames[value]},{ts.tilt_angles[value]},{large_shift_indices[idx]}\n")

        print(f"Performing patch-based alignment on {ts.basename}")
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_patch.com', capture_output=False)
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/align_patch.com', capture_output=False)
        utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)

        known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)
        print(f"Residual error (nm): {resid_error}")

        total_tries = 3
        attempt = 0
        if known_to_unknown > 10 and resid_error >= 1.5:
            print(f"The alignment statistics for {ts.basename} are worse than expected.")
            print(f"Will try to improve the alignments in {total_tries} attempts.")
            while attempt < total_tries:
                print(f"Attempt: {attempt + 1}")
                utils.improve_bad_alignments(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename)
                print(f"Realigning {ts.tilt_dir_name} with a new model as seed.")
                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/align_patch.com', capture_output=False)
                utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)
                attempt += 1
                # Retrieve alignment statistics after improvement
                known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)
                print(f"Residual error (nm): {resid_error}")
                # Break the loop if the condition is no longer met
                if not (known_to_unknown > 10 and resid_error >= 1.5):
                    break
            # Check if attempts were exhausted
            if attempt == total_tries:
                print("Maximum number of realignment attempts reached. \n"
                      "The final alignment accuracy is: \n"
                      f"Residual error (nm): {resid_error} (SD: {sd})")
    else:
        raise FileNotFoundError(f"No valid tilt series found at {ts_path}. Aborting.")


@automateImod.command()
def update_warp_xml(ts_xml_path: str = typer.Option(..., help="Path to Warp processing results"),
                    ts_basename: str = typer.Option(..., help="Basename of the tilt series"),
                    ts_log_path: str = typer.Option(..., help="Path to imod processing directory")):
    """
    Read in the log file generated by align-tilts and disable tilts with large shifts.
    """

    print("Updating the provided XML metadata file")
    md_file = Path(ts_log_path) / f"{ts_basename}/autoImod.marker"
    md = pd.read_csv(md_file, delimiter=',')
    if md_file.exists():
        bad_frames = md["frame_basename"].to_list()
        for bad_frame in bad_frames:
            bad_frame_xml = Path(ts_xml_path) / f"{bad_frame}.xml"
            utils.remove_xml_files(bad_frame_xml)
    else:
        FileNotFoundError(f"{md_file} does not exist.")


@automateImod.command()
def reconstruct_tomograms(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                          ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_"),
                          ts_extension: str = typer.Option(default="mrc",
                                                           help="does the TS end with an st or mrc extension?"),
                          tomo_bin: str = typer.Option(..., help="binned tomogram size")):
    """
    Reconstruct tomograms using IMOD of aligned tilt series for quick viz.
    """
    tomo = pio.Tomogram(path_to_data=ts_data_path, basename=ts_basename, extension=ts_extension, binval=tomo_bin)

    _, pixel_nm, dimX, dimY = pio.read_mrc(f'{tomo.tilt_dir_name}/{tomo.basename}.{tomo.extension}')

    slab_thickness = calc.get_thickness(unbinned_voxel_size=pixel_nm * 10, binval=tomo.binval)

    coms.make_tomogram(tilt_dir_name=tomo.tilt_dir_name, tilt_name=tomo.basename, tilt_extension=tomo.extension,
                       binval=tomo.binval, dimX=dimX, dimY=dimY, thickness=slab_thickness)

    coms.execute_com_file(f'{str(tomo.tilt_dir_name)}/newst_ali.com', )
    coms.execute_com_file(f'{str(tomo.tilt_dir_name)}/tilt_ali.com')
    utils.swap_fast_slow_axes(tomo.tilt_dir_name, tomo.basename)


if __name__ == '__main__':
    automateImod()
