import subprocess

import typer

import automateImod.calc as calc
import automateImod.coms as coms
import automateImod.utils as utils


from pathlib import Path

import automateImod.pio as pio

automateImod = typer.Typer()

"""
Flow -
1. initialise a TS object with user inputs
2. initialse paths to xml and mdoc files which have the same basename as the TS
3. read TS image (this is currently duplicated in utils.remove_tilts_with_large_shifts)
4. Make xcorr_coarse com file and execute it (xcorr_coarse). Make the aligned stack (newst_coarse).
5. Read the .prexg file from step 4. and detect outliers.
    - If outliers are detected 
        Generate a new stack with same name as the original TS name but with the tilts with large shifts removed.
    - If only xml file is provided
        Modify the truth value of the relevant indices with bad tilts
    - If only mdoc file is provided, 
        create a text file with the SubFramePath[idx] of tilts with large shifts
6. Seed patches on the clean coarse algined stack
7. Track patches.
    - If resid_err is worse than 1.5 and the known:unknown ratio is greater than 10, then
        Remove outlier patches using the median value of the badly tracking patches and track with less badly tracking patches
8. Report alignment stats for the aligned tilt series.

TODO:
1. write alignment stats to a text file
2. Not a fan of the output from subprocess command. Perhaps best for me to right my own messages.   
"""


@automateImod.command(no_args_is_help=True)
def align_tilts(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                ts_mdoc_path: Path = typer.Option(None, help="directory containing the tilt series mdoc file"),
                ts_xml_path: Path = typer.Option(None, help="directory containing the tilt series xml file"),
                ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_1"),
                ts_extension: str = typer.Option(default="mrc", help="does the TS end with an st or mrc extension?"),
                ts_tilt_axis: str = typer.Option(..., help="tilt axis value"),
                ts_bin: str = typer.Option(..., help="bin value to reduce the tilt series size by."),
                ts_patch_size: str = typer.Option(..., help="Size of patches to perform patch_tracking")):

    # Initialise a TS object with user inputs
    ts = pio.TiltSeries(path_to_ts_data=ts_data_path, path_to_xml_data=ts_xml_path, path_to_mdoc_data=ts_mdoc_path,
                        basename=ts_basename, extension=ts_extension,
                        tilt_axis_ang=ts_tilt_axis, binval=ts_bin, patch_size=ts_patch_size)
    ts_path = ts.get_mrc_path()
    xml_path = ts.get_xml_path()
    marker_file = ts.tilt_dir_name / ".removed_tilts_with_large_shifts"
    if ts_path.is_file():
        im_data, pixel_nm, dimX, dimY = pio.read_mrc(ts_path)
        # ------ #
        coms.write_xcorr_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, tilt_extension=ts.extension,
                             tilt_axis_ang=ts.tilt_axis_ang)
        coms.make_xcorr_stack_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, tilt_extension=ts.extension,
                                  binval=ts.binval)
        coms.make_patch_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, binval=ts.binval,
                            tilt_axis_ang=ts.tilt_axis_ang, patch=ts.patch_size)
        coms.track_patch_com(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename, pixel_nm=pixel_nm, binval=ts.binval,
                             tilt_axis_ang=ts.tilt_axis_ang, dimX=dimX, dimY=dimY)
        # ------ #

        # ------ #
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_coarse.com', capture_output=False)
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/newst_coarse.com', capture_output=False)
        # ------ #

        if not marker_file.exists():
            print(f"{marker_file} not detected. Continuing with process")
            bad_tilt_indices = utils.detect_large_shifts_afterxcorr(f'{ts.tilt_dir_name}/{ts.basename}.prexg')
            if bad_tilt_indices.size > 0:
                print(f'Detected {len(bad_tilt_indices)} bad tilts in {ts.basename}.')
                print(f'Removing views with large shifts and redoing coarse alignment.')
                utils.remove_tilts_with_large_shifts(ts=ts, im_data=im_data, pixel_nm=pixel_nm, bad_idx=bad_tilt_indices)
                print(f"Redoing coarse alignment with decimated {ts.basename}")

                # ------ #
                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_coarse.com', capture_output=False)
                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/newst_coarse.com', capture_output=False)
                # ------ #

                Path(f"{ts.tilt_dir_name}/.removed_tilts_with_large_shifts").touch()

                if ts_xml_path:
                    print("Updating the provided XML metatdata file")
                    utils.update_xml_files(xml_path, bad_tilt_indices)
        # ------ #
        print(f"Performing patch-based alignment on {ts.basename}.")
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/xcorr_patch.com', capture_output=False)
        coms.execute_com_file(f'{str(ts.tilt_dir_name)}/align_patch.com', capture_output=False)
        utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)
        # ------ #
        known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)
        total_tries = 3
        attempt = 0
        if known_to_unknown > 10 and resid_error >= 1.5:
            print(f"The alignment statistics for {ts.basename} are worse than expected.")
            print(f"Will try to improve the alignments in {total_tries} attempts.")
            while attempt < total_tries:
                print(f"Attempt: {attempt}")
                utils.improve_bad_alignments(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename)
                print(f"Realigning {ts.tilt_dir_name} with a new model as seed.")
                coms.execute_com_file(f'{str(ts.tilt_dir_name)}/align_patch.com', capture_output=False)
                utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)
                attempt += 1
                # Retrieve alignment statistics after improvement
                known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)
                # Break the loop if the condition is no longer met
                if not (known_to_unknown > 10 and resid_error >= 1.5):
                    break
                # Check if attempts were exhausted
            if attempt == total_tries:
                print("Maximum number of realignment attempts reached. \n"
                      "The final alignment accuracy is: \n"
                      f"Residual error (nm): {resid_error} (SD: {sd})")
                #print("Alignment statistics are within acceptable limits. No further action required.")

        else:
            FileNotFoundError(f"File not found at: {ts_path}")

@automateImod.command()
def reconstruct_tomograms(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                          ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_"),
                          ts_extension: str = typer.Option(default="mrc",
                                                           help="does the TS end with an st or mrc extension?"),
                          tomo_bin: str = typer.Option(..., help="binned tomogram size")):

    tomo = pio.Tomogram(path_to_data=ts_data_path, name=ts_basename, extension=ts_extension,binval=tomo_bin)

    _, pixel_nm, dimX, dimY = pio.read_mrc(f'{tomo.tilt_dir_name()}/{tomo.name}.{tomo.extension}')

    slab_thickness = calc.get_thickness(unbinned_voxel_size=pixel_nm*10, binval=tomo.binval)

    coms.make_tomogram(tilt_dir_name=tomo.tilt_dir_name(), tilt_name=tomo.name, binval=tomo.binval, dimX=dimX, dimY=dimY,
                       thickness=slab_thickness)

    coms.execute_com_file(f'{str(tomo.tilt_dir_name())}/newst_ali.com', )
    coms.execute_com_file(f'{str(tomo.tilt_dir_name())}/tilt_ali.com')
    utils.swap_fast_slow_axes(tomo.tilt_dir_name(),tomo.name)

if __name__ == '__main__':
    automateImod()