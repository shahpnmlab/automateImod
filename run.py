import subprocess
import typer

import calc
import coms
import utils
import utils as U
import ps_io
from pathlib import Path

automateImod = typer.Typer()


@automateImod.command()
def black_frames(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                 ts_basename: str = typer.Option(..., help="tilt_series_basename e.g. Position"),
                 ts_extension: str = typer.Option(..., help="does the TS end with an st or mrc "
                                                            "extension?")):
    ts = ps_io.TiltSeries(path_to_data=ts_data_path,
                          name=ts_basename,
                          extension=ts_extension,
                          mdoc=f'{ts_basename}.mdoc',
                          rawtlt=f'{ts_basename}.rawtlt')

    stackfile = ts.path_to_data / ts.name / f'{ts.name}.{ts.extension}'
    mdfile = ts.path_to_data / ts.name / ts.mdoc
    d_idx, d_tilt, d_frames = U.detect_dark_tilts(mrcin=stackfile, mdocf=mdfile)


@automateImod.command()
def align_tilts(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_"),
                ts_extension: str = typer.Option(default="mrc", help="does the TS end with an st or mrc extension?"),
                ts_tilt_axis: str = typer.Option(..., help="tilt axis value"),
                ts_bin: str = typer.Option(..., help="bin value to reduce the tilt series size by."),
                ts_patch_size: str = typer.Option(..., help="Size of patches to perform patch_tracking")):
    ts = ps_io.TiltSeries(path_to_data=ts_data_path, name=ts_basename, extension=ts_extension,
                          mdoc=f'{ts_basename}.mdoc',
                          rawtlt=f'{ts_basename}.rawtlt', tilt_axis_ang=ts_tilt_axis, binval=ts_bin,
                          patch_size=ts_patch_size)

    _, pixel_nm, dimX, dimY = ps_io.read_mrc(f'{ts.tilt_dir_name()}/{ts.name}.{ts.extension}')

    coms.write_xcorr_com(tilt_dir_name=ts.tilt_dir_name(), tilt_name=ts.name, tilt_extension=ts.extension,
                         tilt_axis_ang=ts.tilt_axis_ang)
    coms.make_xcorr_stack_com(tilt_dir_name=ts.tilt_dir_name(), tilt_name=ts.name, tilt_extension=ts.extension,
                              binval=ts.binval)
    coms.make_patch_com(tilt_dir_name=ts.tilt_dir_name(), tilt_name=ts.name, binval=ts.binval,
                        tilt_axis_ang=ts.tilt_axis_ang, patch=ts.patch_size)
    coms.track_patch_com(tilt_dir_name=ts.tilt_dir_name(), tilt_name=ts.name, pixel_nm=pixel_nm, binval=ts.binval,
                         tilt_axis_ang=ts.tilt_axis_ang, dimX=dimX, dimY=dimY)

    cmd_coarse_align_stack = ['submfg', f'{str(ts.tilt_dir_name())}/xcorr_coarse.com']
    cmd_create_coarse_aligned_stack = ['submfg', f'{str(ts.tilt_dir_name())}/newst_coarse.com']
    cmd_make_patch_com = ['submfg', f'{str(ts.tilt_dir_name())}/xcorr_patch.com']
    cmd_track_patches = ['submfg', f'{str(ts.tilt_dir_name())}/align_patch.com']

    coarse_align_stack = subprocess.run(cmd_coarse_align_stack, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
    print(coarse_align_stack.stdout)
    print(coarse_align_stack.stderr)

    U.remove_tilts_with_large_shifts(tilt_dir_name=ts.tilt_dir_name, tilt_name=ts.name, tilt_extension=ts.extension,
                                     tilt_rawtlt=ts.rawtlt)

    make_coarse_align_stack = subprocess.run(cmd_create_coarse_aligned_stack, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE,
                                             text=True)
    print(make_coarse_align_stack.stdout)
    print(make_coarse_align_stack.stderr)

    seed_patches = subprocess.run(cmd_make_patch_com, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
    print(seed_patches.stdout)
    print(seed_patches.stderr)

    track_patches = subprocess.run(cmd_track_patches, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True)
    print(track_patches.stdout)
    print(track_patches.stderr)

    U.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name())
    known_to_unknown, resid_error, sd = U.get_alignment_error(ts.tilt_dir_name())
    while known_to_unknown > 10 and resid_error > 1.5:
        print(f"The alignment statistics for {ts.name} are worse than expected. "
              f"Attempting to improve the alignments by removing badly tracked "
              f"patches.\n"
              f"Ratio of known to unknown: {known_to_unknown}\n"
              f"Residual error (nm): {resid_error} (SD: {sd})")
        U.improve_bad_alignments(tilt_dir_name=ts.tilt_dir_name(), tilt_name=ts.name)
        print("Realigning {} with new model as seed".format(ts.tilt_dir_name()))
        track_patches = subprocess.run(cmd_track_patches, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
        print(track_patches.stdout)
        U.write_ta_coords_log(tilt_dir_name=ts.tilt_dir_name())
        known_to_unknown, resid_error, sd = U.get_alignment_error(ts.tilt_dir_name())
    print("The final alignment accuracy is: \n"
          f"Ratio of known to unknown: {known_to_unknown}\n"
          f"Residual error (nm): {resid_error} (SD: {sd})")


@automateImod.command()
def reconstruct_tomograms(ts_data_path: Path = typer.Option(..., help="directory containing tilt series data"),
                          ts_basename: str = typer.Option(..., help="tilt series_basename e.g. Position_"),
                          ts_extension: str = typer.Option(default="mrc",
                                                           help="does the TS end with an st or mrc extension?"),
                          tomo_bin: str = typer.Option(..., help="binned tomogram size")):

    tomo = ps_io.Tomogram(path_to_data=ts_data_path, name=ts_basename, extension=ts_extension,binval=tomo_bin)

    _, pixel_nm, dimX, dimY = ps_io.read_mrc(f'{tomo.tilt_dir_name()}/{tomo.name}.{tomo.extension}')

    slab_thickness = calc.get_thickness(unbinned_voxel_size=pixel_nm*10, binval=tomo.binval)

    coms.make_tomogram(tilt_dir_name=tomo.tilt_dir_name(), tilt_name=tomo.name, binval=tomo.binval, dimX=dimX, dimY=dimY,
                       thickness=slab_thickness)

    cmd_aligned_stack = ['submfg', f'{str(tomo.tilt_dir_name())}/newst_ali.com']
    cmd_reconstruct_stack = ['submfg', f'{str(tomo.tilt_dir_name())}/tilt_ali.com']


    aligned_stack = subprocess.run(cmd_aligned_stack, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True)
    print(aligned_stack.stdout)
    print(aligned_stack.stderr)

    reconstruct_stack = subprocess.run(cmd_reconstruct_stack, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)
    print(reconstruct_stack.stdout)
    print(reconstruct_stack.stderr)

    utils.swap_fast_slow_axes(tomo.tilt_dir_name(),tomo.name)


if __name__ == '__main__':
    automateImod()

"""
black-frames --ts-data-path example_data/ --ts-basename Position_39_3 --ts-extension mrc

align-tilts --ts-data-path example_data/ --ts-basename Position_39_3 --ts-tilt-axis 84.7 --ts-bin 8 --ts-patch-size 100 --ts-extension mrc

reconstruct-tomograms --ts-data-path example_data/ --ts-basename Position_39_3 --ts-extension mrc --tomo-bin 8
"""
