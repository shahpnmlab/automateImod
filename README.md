# automateImod

An IMOD wrapper for performing automated, parallel, patch-based tilt-series alignment and reconstruction.

This wrapper is designed to streamline the processing of multiple tilt-series by providing a single, powerful command to handle the entire workflow. It is developed with a Warp/Relion/M pipeline in mind.

## INSTALLATION

```git
git clone https://github.com/shahpnmlab/automateImod.git
conda create -n automateImod python=3.10 -y
conda activate automateImod
cd automateImod
pip install -e .
```

Also verify that IMOD is installed in your user account and is in your $PATH. The command `which 3dmod` should return the path to your `3dmod` executable.

Verify the installation by running:
```
automateImod --help
```

## Directory Structure

Before running the program, please ensure your data is organized in the following structure:

```
/path/to/ts_data/
└── TS_BASENAME/
    ├── TS_BASENAME.{mrc,st}
    └── TS_BASENAME.rawtlt

/path/to/MDOCS/
└── TS_BASENAME.{mrc.mdoc,mdoc}
```

Each tilt-series should have its own subdirectory within a main data folder.

## USAGE

The entire alignment, XML update, and reconstruction workflow is handled by the single `run` command. You can process a single tilt-series or a whole folder of them in parallel.

```commandline
automateImod [OPTIONS]
```

### Key Arguments

*   `--ts-data-folder`: **(Required)** The path to the main folder containing your tilt-series subdirectories (e.g., `/path/to/ts_data/`).
*   `--ts-basenames`: A comma-separated list of specific tilt-series basenames to process. If not provided, the program will process all subdirectories in the `--ts-data-folder`.
*   `--n-tasks`: The number of tilt-series to process in parallel. Defaults to 1.
*   `--ts-tilt-axis`: **(Required)** The tilt axis angle.
*   `--ts-patch-size`: **(Required)** The size of the patches for tracking, in Angstroms.
*   `--is-warp-proj`: A flag to indicate that this is a Warp project. If set, the program will update the Warp XML and .tomostar files after alignment.
*   `--reconstruct`: A flag to trigger tomogram reconstruction after alignment.
*   `--ts-xml-path`: Path to the folder containing Warp XML files (required if `--is-warp-proj` is set).
*   `--ts-tomostar-path`: Path to the folder containing TomoStar files (required if `--is-warp-proj` is set).
*   `--tomo-bin`: The binning factor for the reconstructed tomogram (required if `--reconstruct` is set).

### Example: Processing multiple tilt-series in parallel

To process all tilt-series in `/path/to/ts_data` using 4 parallel workers, and also reconstruct binned tomograms:

```bash
automateImod \
    --ts-data-folder /path/to/ts_data/ \
    --ts-mdoc-path /path/to/MDOCS/ \
    --ts-tilt-axis 84.7 \
    --ts-patch-size 350 \
    --n-tasks 4 \
    --reconstruct \
    --tomo-bin 4
```

### Progress Bar

The program features a detailed progress bar that shows the status of each processing step for each tilt-series when running in parallel. This gives you a real-time view of the alignment progress.


# Changelog

### v0.6.0
- **Parallel Processing**: Integrated `dask.distributed` to enable robust parallel processing of multiple tilt-series.
- **Granular Progress Bar**: Implemented a detailed progress bar that shows the real-time status of each alignment step for each tilt-series.
- **Enhanced Logging**: Improved logging to capture subprocess outputs for better debugging.

### v0.5.0
- **Unified Pipeline**: Combined `align-tilts`, `update-warp-xml`, and `reconstruct-tomograms` into a single, unified `run` command.
- **Initial simple dask parallelisation**: implemented and tested, controlled by the `--n-tasks` argument.

### v0.4.5
- Migrated packaging to `pyproject.toml`.
- Fixed a bug in `update-warp-xml` that caused incomplete removal of bad tilts from `.tomostar` files.
- Refactored the `detect_large_shifts_afterxcorr` function for more robust outlier detection.
- Updated documentation for new command-line arguments.

### v0.4.3
- Introduction of `--min-fov` arg for more consistent removal of aberrant views.
- Relaxed requirement for `.mdoc` files extension to include `.mrc.mdoc` as well.
- Richer `TiltSeries` class

### v0.4.2
- Updates warp xml files by changing the `<frame>.xml` to `<frame>.xml.bkp`

### v0.4.1
- Introduced a dark tilt detection routine to remove these tilts even before performing coarse alignments.
- Automatic detection of tilt series extension.

# TODO
- Support Relion 5.0 star files at some point.
- Abort alignments if the alignments stats are bad even after throwing out badly tracking tilts.
