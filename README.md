# automateImod

An IMOD wrapper for performing patch-based tilt-series alignment.
This wrapper will continue being developed as and when I feel a certain functionality is needed.
Also note that I am developing this against Warp/Relion/M pipeline. If I start using Relion 5.0 or < I might try to
develop for it as well. However, dont hold your breath.

##I NSTALLATION

```
git clone https://github.com/shahpnmlab/automateImod.git
conda create -n automateImod python=3.10 -y
conda activate automateImod
cd automateImod
pip install -r requirements.txt
pip install -e .
```

Also verify that IMOD is installed in your user account and installed and in your ```$PATH```; ```which 3dmod```should
tell you the path to your 3dmod executable.

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
  align-tilts
  dark-frames-remover
  reconstruct-tomograms
```

## USAGE

Before staring to use the program make sure that the expected directory structure is followed, which is -

```commandline
/PATH/TO/TS_BASENAME/TS_BASENAME.{mrc,st}
                     TS_BASENAME.rawtlt
                     TS_BASENAME.mdoc
```

The main sub-command that you will want to run is the ```align-tilts``` command

```commandline
automateImod align-tilts --help

 automateImod align-tilts [OPTIONS]

Options:
  --ts-data-path PATH   directory containing tilt series data  [required]
  --ts-basename TEXT    tilt series_basename e.g. Position_  [required]
  --ts-extension TEXT   does the TS end with an st or mrc extension?
                        [default: mrc]
  --ts-tilt-axis TEXT   tilt axis value  [required]
  --ts-bin TEXT         bin value to reduce the tilt series size by.
                        [required]
  --ts-patch-size TEXT  Size of patches to perform patch_tracking  [required]
  --help                Show this message and exit.
```

To align multiple series, run it as a bash ```for``` loop, by looping over folders within which the tilt-series data
resides.

The mdoc file is presently optional to include, but in the future, it will be used to identify
dark frames, as well frames that shift too much during the tilt series alignment process.

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

##TODO

- Detect dark frames and note which ones those are in a log file by parsing the mdoc file.
- Log tilts that move more than the hard-coded sDev and log the tilt names to a logfile.