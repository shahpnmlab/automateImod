import subprocess
import logging

def write_xcorr_com(tilt_dir_name, tilt_name, tilt_extension, tilt_axis_ang):
    """
    coarse_align
    """
    with open(f"{tilt_dir_name}/xcorr_coarse.com", "w") as fout:
        fout.write(
            "# THIS IS A COMMAND FILE TO RUN TILTXCORR AND DETERMINE CROSS-CORRELATION\n")
        fout.write("# ALIGNMENT OF A TILT SERIES\n")
        fout.write("# TO RUN TILTXCORR\n")
        fout.write("####CreatedVersion####4.11.1\n")
        fout.write("#\n")
        fout.write(
            "# Add BordersInXandY to use a centered region smaller than the default\n")
        fout.write(
            "# or XMinAndMax and YMinAndMax  to specify a non-centered region\n")
        fout.write("#\n")
        fout.write("$tiltxcorr -StandardInput\n")
        fout.write(f"InputFile\t{tilt_dir_name}/{tilt_name}.{tilt_extension}\n")
        fout.write(f"OutputFile\t{tilt_dir_name}/{tilt_name}.prexf\n")
        fout.write(f"TiltFile\t{tilt_dir_name}/{tilt_name}.rawtlt\n")
        fout.write(f"RotationAngle\t{tilt_axis_ang}\n")
        fout.write("FilterSigma1\t0.03\n")
        fout.write("FilterRadius2\t0.25\n")
        fout.write("FilterSigma2\t0.05\n")
        fout.write("$if (-e ./savework) ./savework")


def make_xcorr_stack_com(tilt_dir_name, tilt_name, tilt_extension, binval):
    """
    coarse_align_stack
    """
    with open(f"{tilt_dir_name}/newst_coarse.com", "w") as fout:
        fout.write("# THIS IS A COMMAND FILE TO PRODUCE A PRE-ALIGNED STACK\n")
        fout.write("#\n")
        fout.write(
            "# The stack will be floated and converted to bytes under the assumption that\n")
        fout.write(
            "# you will go back to the raw stack to make the final aligned stack\n")
        fout.write("#\n")
        fout.write("$xftoxg\n")
        fout.write("0\n")
        fout.write(f"{tilt_dir_name}/{tilt_name}.prexf\n")
        fout.write(f"{tilt_dir_name}/{tilt_name}.prexg\n")
        fout.write("$newstack -StandardInput\n")
        fout.write(f"InputFile\t{tilt_dir_name}/{tilt_name}.{tilt_extension}\n")
        fout.write(f"OutputFile\t{tilt_dir_name}/{tilt_name}_preali.mrc\n")
        fout.write(f"TransformFile\t{tilt_dir_name}/{tilt_name}.prexg\n")
        fout.write("FloatDensities\t2\n")
        fout.write(f"BinByFactor\t{binval}\n")
        fout.write("#DistortionField\t.idf\n")
        fout.write("ImagesAreBinned\t1.0\n")
        fout.write("AntialiasFilter\t5\n")
        fout.write(f"#GradientFile\t{tilt_name}.maggrad\n")
        fout.write("$if (-e ./savework) ./savework\n")


def make_patch_com(tilt_dir_name, tilt_name, binval, tilt_axis_ang, patch):
    """
    seed_patches
    """
    with open(f"{tilt_dir_name}/xcorr_patch.com", "w") as fout:
        fout.write("$goto doxcorr\n")
        fout.write("$doxcorr:\n")
        fout.write("$tiltxcorr -StandardInput\n")
        fout.write("BordersInXandY\t82,82\n")
        fout.write("IterateCorrelations\t2\n")
        fout.write(f"ImagesAreBinned\t {binval}\n")
        fout.write(f"InputFile\t{tilt_dir_name}/{tilt_name}_preali.mrc\n")
        fout.write(f"OutputFile\t{tilt_dir_name}/{tilt_name}_pt.fid\n")
        fout.write(f"PrealignmentTransformFile\t{tilt_dir_name}/{tilt_name}.prexg\n")
        fout.write(f"TiltFile\t{tilt_dir_name}/{tilt_name}.rawtlt\n")
        fout.write(f"RotationAngle\t{tilt_axis_ang}\n")
        fout.write("FilterSigma1\t0.03\n")
        fout.write("FilterRadius2\t0.25\n")
        fout.write("FilterSigma2\t0.05\n")
        fout.write(f"SizeOfPatchesXandY\t{patch},{patch}\n")
        fout.write("OverlapOfPatchesXandY\t0.8,0.8\n")
        fout.write("$dochop:\n")
        fout.write("$imodchopconts -StandardInput\n")
        fout.write(f"InputModel {tilt_dir_name}/{tilt_name}_pt.fid\n")
        fout.write(f"OutputModel {tilt_dir_name}/{tilt_name}.fid\n")
        fout.write("MinimumOverlap\t4\n")
        fout.write("AssignSurfaces 1\n")
        fout.write("$if (-e ./savework) ./savework\n")


def track_patch_com(tilt_dir_name, tilt_name, pixel_nm, binval, tilt_axis_ang, dimX, dimY):
    """
    track_patches
    """
    with open(f"{tilt_dir_name}/align_patch.com", "w") as fout:
        fout.write("# THIS IS A COMMAND FILE TO RUN TILTALIGN\n")
        fout.write("#\n")
        fout.write("####CreatedVersion####4.11.1\n")
        fout.write('#\n')
        fout.write("# To exclude views, add a line ExcludeList view_list with the list of views\n")
        fout.write("#\n")
        fout.write("# To specify sets of views to be grouped separately in automapping, add a line\n")
        fout.write("# SeparateGroup view_list with the list of views, one line per group\n")
        fout.write("#\n")
        fout.write("$tiltalign -StandardInput\n")
        fout.write("RobustFitting\n")
        fout.write("WeightWholeTracks\n")
        fout.write(f"ModelFile\t{tilt_dir_name}/{tilt_name}.fid\n")
        fout.write(f"ImageFile\t{tilt_dir_name}/{tilt_name}_preali.mrc\n")
        fout.write(f"#ImageSizeXandY\t{dimX},{dimY}\n")
        fout.write(f"ImagesAreBinned\t{binval}\n")
        fout.write(f"OutputModelFile \t{tilt_dir_name}/{tilt_name}.3dmod\n")
        fout.write(f"OutputResidualFile\t{tilt_dir_name}/{tilt_name}.resid\n")
        fout.write(f"OutputFidXYZFile\t{tilt_dir_name}/{tilt_name}fid.xyz\n")
        fout.write(f"OutputTiltFile \t{tilt_dir_name}/{tilt_name}.tlt\n")
        fout.write(f"OutputXAxisTiltFile {tilt_dir_name}/{tilt_name}.xtilt\n")
        fout.write(f"OutputTransformFile {tilt_dir_name}/{tilt_name}.tltxf\n")
        fout.write(f"OutputFilledInModel\t{tilt_dir_name}/{tilt_name}_nogaps.fid\n")
        fout.write(f"RotationAngle\t{tilt_axis_ang}\n")
        fout.write(f"UnbinnedPixelSize\t{pixel_nm}\n")
        fout.write(f"TiltFile\t{tilt_dir_name}/{tilt_name}.rawtlt\n")
        fout.write("#\n")
        fout.write("# ADD a recommended tilt angle change to the existing AngleOffset value\n")
        fout.write("#\n")
        fout.write("AngleOffset 0.0\n")
        fout.write("RotOption  -1\n")
        fout.write("RotDefaultGrouping 5\n")
        fout.write("#\n")
        fout.write("# TiltOption 0 fixes tilts, 2 solves for all tilt angles; change to 5 to solve\n")
        fout.write("# for fewer tilts by grouping views by the amount in TiltDefaultGrouping\n")
        fout.write("#\n")
        fout.write("TiltOption 0\n")
        fout.write("TiltDefaultGrouping 5\n")
        fout.write("MagReferenceView\t1\n")
        fout.write("MagOption\t0\n")
        fout.write("MagDefaultGrouping  4\n")
        fout.write("#\n")
        fout.write("# To solve for distortion, change both XStretchOption and SkewOption to 3;\n")
        fout.write("# to solve for skew only leave XStretchOption at 0\n")
        fout.write("#\n")
        fout.write("XStretchOption \t0\n")
        fout.write("SkewOption  0\n")
        fout.write("XStretchDefaultGrouping \t7\n")
        fout.write("SkewDefaultGrouping 11\n")
        fout.write("BeamTiltOption \t0\n")
        fout.write("#\n")
        fout.write("# To solve for X axis tilt between two halves of a dataset, set XTiltOption to 4\n")
        fout.write("#\n")
        fout.write("XTiltOption 0\n")
        fout.write("XTiltDefaultGrouping\t2000\n")
        fout.write("#\n")
        fout.write("# Criterion # of S.D's above mean residual to report (- for local mean)\n")
        fout.write("#\n")
        fout.write("ResidualReportCriterion \t3.0\n")
        fout.write("SurfacesToAnalyze  1\n")
        fout.write("MetroFactor 0.25\n")
        fout.write("MaximumCycles \t1000\n")
        fout.write("KFactorScaling \t1.0\n")
        fout.write("NoSeparateTiltGroups \t1\n")
        fout.write("#\n")
        fout.write("# ADD a recommended amount to shift up to the existing AxisZ")
        fout.write("#\n")
        fout.write("AxisZShift 0.0\n")
        fout.write("ShiftZFromOriginal      1\n")
        fout.write("#\n")
        fout.write("# Set to 1 to do local alignments\n")
        fout.write("#\n")
        fout.write("LocalAlignments \t0\n")
        fout.write("OutputLocalFile \t" + tilt_name + "local.xf\n")
        fout.write("#\n")
        fout.write("# Target size of local patches to solve for in X and Y\n")
        fout.write("#\n")
        fout.write("TargetPatchSizeXandY\t700,700\n")
        fout.write("MinSizeOrOverlapXandY\t0.5,0.5\n")
        fout.write("#\n")
        fout.write(
            "# Minimum fiducials total and on one surface if two surfaces\n")
        fout.write("#\n")
        fout.write("MinFidsTotalAndEachSurface\t8,3\n")
        fout.write("FixXYZCoordinates\t0\n")
        fout.write("LocalOutputOptions\t1,0,1\n")
        fout.write("LocalRotOption \t3\n")
        fout.write("LocalRotDefaultGrouping \t6\n")
        fout.write("LocalTiltOption \t5\n")
        fout.write("LocalTiltDefaultGrouping\t6\n")
        fout.write("LocalMagReferenceView   \t1\n")
        fout.write("LocalMagOption  \t3\n")
        fout.write("LocalMagDefaultGrouping \t7\n")
        fout.write("LocalXStretchOption 0\n")
        fout.write("LocalXStretchDefaultGrouping     7\n")
        fout.write("LocalSkewOption     0\n")
        fout.write("LocalSkewDefaultGrouping   11\n")
        fout.write("#\n")
        fout.write("# COMBINE TILT TRANSFORMS WITH PREALIGNMENT TRANSFORMS\n")
        fout.write("#\n")
        fout.write("$xfproduct -StandardInput\n")
        fout.write(f"InputFile1 {tilt_dir_name}/{tilt_name}.prexg\n")
        fout.write(f"InputFile2 {tilt_dir_name}/{tilt_name}.tltxf\n")
        fout.write(f"OutputFile {tilt_dir_name}/{tilt_name}_fid.xf\n")
        fout.write("ScaleShifts 1.0,5.0\n")
        fout.write(f"$b3dcopy -p {tilt_dir_name}/{tilt_name}_fid.xf {tilt_dir_name}/{tilt_name}.xf\n")
        fout.write(f"$b3dcopy -p {tilt_dir_name}/{tilt_name}.tlt {tilt_dir_name}/{tilt_name}_fid.tlt\n")
        fout.write("#\n")
        fout.write("# CONVERT RESIDUAL FILE TO MODEL\n")
        fout.write("#\n")
        fout.write(
            f"$if (-e {tilt_dir_name}/{tilt_name}.resid) patch2imod -s 10 {tilt_dir_name}/{tilt_name}.resid {tilt_dir_name}/{tilt_name}.resmod\n")
        fout.write("$if (-e ./savework) ./savework\n")


def make_tomogram(tilt_dir_name, tilt_name, tilt_extension, dimX, dimY, binval, thickness):
    dimX = int(dimX)
    dimY = int(dimY)
    binval = int(binval)

    tilt_dir_name = str(tilt_dir_name)
    bin_x = int(dimX / binval)
    bin_y = int(dimY / binval)

    with open(tilt_dir_name + "/" + "newst_ali.com", "w") as fout:
        fout.write(f'# The offset argument should be 0,0 for no offset, 0,300 to take an area\n')
        fout.write(f'# 300 pixels above the center, etc.\n')
        fout.write(f'#\n')
        fout.write(f'$newstack -StandardInput\n')
        fout.write(f'InputFile   {tilt_dir_name}/{tilt_name}.{tilt_extension}\n')
        fout.write(f'OutputFile  {tilt_dir_name}/{tilt_name}.ali\n')
        fout.write(f'TransformFile   {tilt_dir_name}/{tilt_name}.xf\n')
        fout.write(f'TaperAtFill 1,0\n')
        fout.write(f'AdjustOrigin\n')
        fout.write(f'SizeToOutputInXandY {bin_x},{bin_y}\n')
        fout.write(f'OffsetsInXandY  0.0,0.0\n')
        fout.write(f'#DistortionField    .idf\n')
        fout.write(f'ImagesAreBinned 1.0\n')
        fout.write(f'BinByFactor {binval}\n')
        fout.write(f'AntialiasFilter -1\n')
        fout.write(f'#GradientFile   {tilt_dir_name}/{tilt_name}.maggrad\n')
        fout.write(f'$if (-e ./savework) ./savework\n')

    with open(tilt_dir_name + "/" + "tilt_ali.com", "w") as fout:
        fout.write(f'# Command file to run Tilt\n')
        fout.write(f'#\n')
        fout.write(f'####CreatedVersion####4.11.1\n')
        fout.write(f'#\n')
        fout.write(f'# RADIAL specifies the frequency at which the Gaussian low pass filter begins\n')
        fout.write(f'#   followed by the standard deviation of the Gaussian roll-off\n')
        fout.write(f'#\n')
        fout.write(f'# LOG takes the logarithm of tilt data after adding the given value\n')
        fout.write(f'#\n')
        fout.write(f'$tilt -StandardInput\n')
        fout.write(f'InputProjections {tilt_dir_name}/{tilt_name}.ali\n')
        fout.write(f'OutputFile {tilt_dir_name}/{tilt_name}.rec\n')
        fout.write(f'IMAGEBINNED 1\n')
        fout.write(f'TILTFILE {tilt_dir_name}/{tilt_name}.tlt\n')
        fout.write(f'UseGPU 0\n')
        fout.write(f'ActionIfGPUFails 0,0\n')
        fout.write(f'THICKNESS {thickness}\n')
        fout.write(f'RADIAL 0.35 0.035\n')
        fout.write(f'FalloffIsTrueSigma 1\n')
        fout.write(f'XAXISTILT 0.0\n')
        fout.write(f'LOG 0.0\n')
        fout.write(f'SCALE 0.0 250.0\n')
        fout.write(f'PERPENDICULAR\n')
        fout.write(f'MODE 2\n')
        fout.write(f'FULLIMAGE {bin_x} {bin_y}\n')
        fout.write(f'SUBSETSTART 0 0\n')
        fout.write(f'AdjustOrigin\n')
        fout.write(f'OFFSET 0.0\n')
        fout.write(f'SHIFT 0.0 0.0\n')
        fout.write(f'XTILTFILE {tilt_dir_name}/{tilt_name}.xtilt\n')
        fout.write(f'FakeSIRTiterations 20\n')
        fout.write(f'$if (-e ./savework) ./savework\n')


def execute_com_file(path_to_com_file, additional_args=None, capture_output=True):
        """
        Executes a given command using subprocess.

        Args:
            path_to_com_file (str): self-explanatory
            additional_args (list, optional): Additional arguments for the command.
            capture_output (bool, optional): Whether to capture output or not.

        Returns:
            str: The output of the subprocess command, if captured.
        """
        try:
            cmd = ["submfg", f'{path_to_com_file}']
            if additional_args:
                cmd.extend(additional_args)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            if capture_output:
                print(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running subprocess command: {e}")
            print(f"Error: {e.stderr}")