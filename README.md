# automateImod

An IMOD wrapper for performing patch-based tilt-series alignment.
This wrapper will continue being developed as and when I feel a certain functionality is needed.
Also note that I am developing this against Warp/Relion/M pipeline. If I start using Relion 5.0 or < I might try to
develop for it as well. However, dont hold your breath.

## INSTALLATION

```
git clone https://github.com/shahpnmlab/automateImod.git
conda create -n automateImod python=3.10 -y
conda activate automateImod
cd automateImod
pip install -e .
```

Also verify that IMOD is installed in your user account and installed and in your ```$PATH```; ```which 3dmod```should
tell you the path to your ```3dmod``` executable.

Verify if the package has been installed using the following command
```automateImod --help```
You should see -

```commandline
Usage: automateImod [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion [bash|zsh|fish|powershell|pwsh]
                                  Install completion for the specified shell.
  --show-completion [bash|zsh|fish|powershell|pwsh]
                                  Show completion for the specified shell, to
                                  copy it or customize the installation.
  --help                          Show this message and exit.

Commands:
  align-tilts            Perform patch-based tilt series tracking using...
  reconstruct-tomograms  Reconstruct tomograms using IMOD of aligned tilt...
  update-warp-xml        Read in the log file generated by align-tilts...

```

## USAGE

Before staring to use the program make sure that the expected directory structure is followed, which is -

```commandline
/path/to/IMOD/
└── TS_BASENAME/
    ├── TS_BASENAME.{mrc,st}
    └── TS_BASENAME.rawtlt

/path/to/MDOCS/
└── TS_BASENAME.{mrc.mdoc,mdoc}
```

The main sub-command that you will want to run is the ```align-tilts``` command

```commandline
python -m automateImod.run align-tilts [OPTIONS]

  Perform patch-based tilt series tracking using IMOD routines

Options:
  --ts-basename TEXT       tilt series_basename e.g. Position_1  [required]
  --ts-data-path PATH      directory containing tilt series data  [required]
  --ts-mdoc-path PATH      directory containing the tilt series mdoc file
  --ts-tomostar-path PATH  directory containing the tomostar file
  --ts-tilt-axis TEXT      tilt axis value  [required]
  --ts-bin TEXT            bin value to reduce the tilt series size by.
                           [default: 1]
  --ts-patch-size TEXT     Size of patches to perform patch_tracking (Å)
                           [required]
  --min-fov FLOAT          Minimum required field of view  [default: 0.7]
  --max-attempts INTEGER   How many attempts before quitting refinement
                           [default: 3]
  --help                   Show this message and exit.


```

To align multiple series, run it as a bash ```for``` loop, by looping over folders within which the tilt-series data
resides.

The mdoc file will be read to identify problematic tilt-frames and logged to text file in the imod processing directory. 

The ```update-wrp-xml``` command can then be issued to disable tilts with large shifts.
```commandline
automateImod update-warp-xml --help
Usage: automateImod update-warp-xml [OPTIONS]

  Read in the log file generated by align-tilts and disable tilts with large
  shifts.

Options:
  --ts-xml-path TEXT  Path to Warp processing results  [required]
  --ts-basename TEXT  Basename of the tilt series  [required]
  --ts-log-path TEXT  Path to imod processing directory  [required]
  --help              Show this message and exit.

```
You can briefly review the results of your aligned tilt-series by reconstructing tomograms by running -

```commandline
Usage: automateImod reconstruct-tomograms [OPTIONS]

Options:
  --ts-data-path PATH  directory containing tilt series data  [required]
  --ts-basename TEXT   tilt series_basename e.g. Position_  [required]
  --ts-extension TEXT  does the TS end with an st or mrc extension?  [default:
                       mrc]
  --tomo-bin TEXT      binned tomogram size  [required]
  --help               Show this message and exit.

```

# Changelog 
### v0.4.3
- Introduction of `--min_fov` arg for more consistent removal of aberrant views (à la
[WarpTools](https://github.com/warpem/warp/tree/main)).
- Relaxed requirement for `.mdoc` files extension to include `.mrc.mdoc` as well.
- Richer `TiltSeries` class

### v0.4.2
- Updates warp xml files by changing the \<frame\>.xml to \<frame\>.xml.bkp
### v0.4.1
- Introduced a dark tilt detection routine to remove these tilts even before performing coarse alignments.
- Automatic detection of tilt series extension.


# TODO
- Support Relion 5.0 star files at some point.
- The automateImod.marker file has the wrong tilt indices.
- Abort alignments if the alignments stats are bad even after throwing out badly tracking tilts
- Save the outputs to log file