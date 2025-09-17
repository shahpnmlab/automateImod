import mrcfile
import typer
import starfile
import shutil  # Add this if not present

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


@automateImod.command(no_args_is_help=True)
def align_tilts(
    ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_1"),
    ts_data_path: Path = typer.Option(
        ..., help="directory containing tilt series data"
    ),
    ts_mdoc_path: Path = typer.Option(
        None, help="directory containing the tilt series mdoc file"
    ),
    ts_tomostar_path: Path = typer.Option(
        None, help="directory containing the tomostar file"
    ),
    ts_tilt_axis: str = typer.Option(..., help="tilt axis value"),
    ts_bin: str = typer.Option(1, help="bin value to reduce the tilt series size by."),
    ts_patch_size: str = typer.Option(
        ..., help="Size of patches to perform patch_tracking (Å)"
    ),
    min_fov: float = typer.Option(0.7, help="Minimum required field of view"),
    max_attempts: int = typer.Option(
        3, help="How many attempts before quitting refinement"
    ),
    max_shift_nm: float = typer.Option(
        None,
        help="Maximum acceptable absolute shift in nm (default: 0.2 * min(width_nm, height_nm))",
    ),
    max_shift_rate: float = typer.Option(
        50.0, help="Maximum acceptable shift rate in nm/degree (default: 50.0)"
    ),
    use_statistical_analysis: bool = typer.Option(
        True,
        help="Whether to use statistical outlier detection for shifts (default: True)",
    ),
):
    """
    Perform patch-based tilt series tracking using IMOD routines
    """

    # Initialise a TS object with user inputs
    ts = pio.TiltSeries(
        path_to_ts_data=ts_data_path,
        path_to_mdoc_data=ts_mdoc_path,
        path_to_tomostar=ts_tomostar_path,
        basename=ts_basename,
        tilt_axis_ang=ts_tilt_axis,
        binval=ts_bin,
        patch_size_ang=ts_patch_size,  # Now in angstroms
    )

    ts_path = ts.get_mrc_path()
    marker_file = ts.tilt_dir_name / "autoImod.marker"

    if ts_path.is_file():
        # im_data, pixel_nm, dimX, dimY = pio.read_mrc(ts_path)
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
            pixel_nm=ts.pixel_size / 10,  # for nm
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
            print(f"Marker file not detected in {marker_file.parent}\nProcessing...")
            print("Looking for dark tilts...")

            dark_frame_indices = utils.detect_dark_tilts(
                ts_data=im_data, ts_tilt_angles=ts.tilt_angles
            )

            if len(dark_frame_indices) > 0:
                print(f"Detected {len(dark_frame_indices)} dark tilts in {ts.basename}")
                print(f"Removing dark tilts...")
                utils.remove_bad_tilts(
                    ts=ts,
                    im_data=im_data,
                    pixel_nm=ts.pixel_size / 10,
                    bad_idx=dark_frame_indices,
                )
                ts.removed_indices.extend(dark_frame_indices)
                del im_data
                im_data, pixel_nm, dimX, dimY = pio.read_mrc(ts_path)
            else:
                print(
                    "No dark frames found in the current tilt series. Proceeding with coarse alignments..."
                )

            # Write marker file (even if no dark frames were detected)
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
                min_fov_fraction=min_fov,
                max_shift_nm=max_shift_nm,
                max_shift_rate=max_shift_rate,
                use_statistical_analysis=use_statistical_analysis,
            )

            if len(large_shift_indices) > 0:
                print(
                    f"Detected {len(large_shift_indices)} badly tracking tilts in {ts.basename}"
                )
                print(f"Removing badly tracked tilts...")

                # Convert decimated stack indices to original stack indices
                original_large_shift_indices = [
                    np.where(np.array(original_tilt_angles) == ts.tilt_angles[idx])[0][
                        0
                    ]
                    for idx in large_shift_indices
                ]

                utils.remove_bad_tilts(
                    ts=ts,
                    im_data=im_data,
                    pixel_nm=ts.pixel_size / 10,
                    bad_idx=large_shift_indices,
                )
                ts.removed_indices.extend(original_large_shift_indices)
                print(f"Redoing coarse alignment with decimated {ts.basename} stack")

                coms.execute_com_file(
                    f"{str(ts.tilt_dir_name)}/xcorr_coarse.com", capture_output=False
                )
                coms.execute_com_file(
                    f"{str(ts.tilt_dir_name)}/newst_coarse.com", capture_output=False
                )

                # Append large shift frames to marker file
                with open(marker_file, "a") as fout:
                    for idx in original_large_shift_indices:
                        if idx < len(original_tilt_angles) and idx < len(
                            original_tilt_frames
                        ):
                            frame_name = original_tilt_frames[idx]
                            fout.write(
                                f"{frame_name},{original_tilt_angles[idx]},{idx}\n"
                            )
                        else:
                            print(
                                f"Warning: Index {idx} is out of range for original tilt_angles or tilt_frames."
                            )

        # After all tilt removals, ensure the removed_indices are unique and sorted
        ts.removed_indices = sorted(set(ts.removed_indices))

        print(f"Performing patch-based alignment on {ts.basename}")
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/xcorr_patch.com", capture_output=False
        )
        coms.execute_com_file(
            f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
        )
        utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)

        known_to_unknown, resid_error, sd = utils.get_alignment_error(ts.tilt_dir_name)

        if resid_error is not None:
            print(f"Residual error (nm): {resid_error}")
        else:
            print(
                "Could not retrieve alignment statistics. The alignment may have failed."
            )
            return

        total_tries = max_attempts
        attempt = 0
        if (
            known_to_unknown is not None
            and resid_error is not None
            and known_to_unknown > 10
            and resid_error >= 1.5
        ):
            print(
                f"The alignment statistics for {ts.basename} are worse than expected."
            )
            print(f"Will try to improve the alignments in {total_tries} attempts.")
            while attempt < total_tries:
                print(f"Attempt: {attempt + 1}")
                utils.improve_bad_alignments(
                    tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.basename
                )
                print(f"Realigning {ts.tilt_dir_name} with a new model as seed.")
                coms.execute_com_file(
                    f"{str(ts.tilt_dir_name)}/align_patch.com", capture_output=False
                )
                utils.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name)
                attempt += 1
                known_to_unknown, resid_error, sd = utils.get_alignment_error(
                    ts.tilt_dir_name
                )
                if resid_error is not None:
                    print(f"Residual error (nm): {resid_error}")
                else:
                    print("Could not retrieve alignment statistics.")
                if not (
                    known_to_unknown is not None
                    and resid_error is not None
                    and known_to_unknown > 10
                    and resid_error >= 1.5
                ):
                    break
            if attempt == total_tries:
                print("Maximum number of realignment attempts reached.")

        if resid_error is not None:
            print(f"The final alignment accuracy is:")
            print(f"Residual error (nm): {resid_error} (SD: {sd})")
            print(f"Building an aligned stack for {ts.basename}..-")
            coms.execute_com_file(
                f"{str(ts.tilt_dir_name)}/newst_ali.com", capture_output=False
            )
        else:
            print("Could not retrieve final alignment statistics.")
    else:
        raise FileNotFoundError(f"No valid tilt series found at {ts_path}. Aborting.")


@automateImod.command()
def update_warp_xml(
    ts_basename: str = typer.Option(..., help="Basename of the tilt series"),
    ts_xml_path: str = typer.Option(..., help="Path to Warp processing results"),
    ts_tomostar_path: str = typer.Option(..., help="Path to tomostar file"),
    ts_log_path: str = typer.Option(..., help="Path to imod processing directory"),
    backup_xml: bool = typer.Option(
        True,
        "--backup-xml/--no-backup-xml",
        help="Create a backup of the XML file before updating. Defaults to True.",
    ),  # New option
):
    """
    Read in the log file generated by align-tilts, remove tilts with large shifts from the XML file,
    and update the tomostar file.
    """
    print("Updating the provided XML metadata file and tomostar file")

    # Read the autoImod.marker file
    marker_file = Path(ts_log_path) / f"{ts_basename}/autoImod.marker"
    if marker_file.exists():
        md = pd.read_csv(marker_file, delimiter=",")
        bad_frames = md["frame_basename"].tolist()
    else:
        print(f"Warning: {marker_file} does not exist.")
        return

    # Update XML file
    xml_file = Path(ts_xml_path) / f"{ts_basename}.xml"
    if not xml_file.exists():
        print(f"Error: XML file {xml_file} does not exist.")
        return

    # New backup logic
    if backup_xml:
        backup_file_path = xml_file.with_suffix(
            xml_file.suffix + ".bak"
        )  # e.g., file.xml.bak
        try:
            shutil.copy2(xml_file, backup_file_path)
            print(f"Backup of {xml_file} created at {backup_file_path}")
        except Exception as e:
            print(f"Error creating backup for {xml_file}: {e}")
            # Decide if you want to proceed without backup or stop. For now, just print error and continue.

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
            element.text = "\n" + "\n".join(updated_lines) + "\n"  # Preserve formatting
        else:
            print(f"Warning: {element_name} element not found in the XML file.")

    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    print(f"Updated XML file: {xml_file}")

    # Update tomostar file
    tomostar_file = Path(ts_tomostar_path) / f"{ts_basename}.tomostar"
    if not tomostar_file.exists():
        print(f"Error: Tomostar file {tomostar_file} does not exist.")
        return

    tomostar_data = starfile.read(tomostar_file)

    good_mask = (
        ~tomostar_data["wrpMovieName"].apply(lambda x: Path(x).name).isin(bad_frames)
    )
    updated_tomostar_data = tomostar_data[good_mask]

    starfile.write(updated_tomostar_data, tomostar_file, overwrite=True)
    print(f"Updated tomostar file: {tomostar_file}")


@automateImod.command()
def reconstruct_tomograms(
    ts_data_path: Path = typer.Option(
        ..., help="directory containing tilt series data"
    ),
    ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_"),
    ts_extension: str = typer.Option(
        default="mrc", help="does the TS end with an st or mrc extension?"
    ),
    tomo_bin: str = typer.Option(..., help="binned tomogram size"),
):
    """
    Reconstruct tomograms using IMOD of aligned tilt series for quick viz.
    """
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


if __name__ == "__main__":
    automateImod()

# align-tilts --ts-basename lam13_pos02_ts_001 --ts-data-path tiltseries/tiltstack/ --ts-mdoc-path mdocs/  --ts-tilt-axis 84.7 --ts-bin 1 --ts-patch-size 2000
# update-warp-xml --ts-basename lam13_pos02_ts_001 --ts-xml-path tiltseries/ --ts-tomostar-path tomostar/ --ts-log-path tiltseries/tiltstack/
