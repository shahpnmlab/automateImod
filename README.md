# automateImod

An IMOD wrapper for performing patch-based tilt-series alignment with parallel processing capabilities.

This wrapper automates the alignment of cryo-electron tomography tilt series using IMOD's patch-based tracking. It's primarily developed for use with the Warp/Relion/M pipeline but may work with other workflows.

## Features

- Automated patch-based tilt series alignment
- Automatic detection and removal of dark/low-quality tilts
- Detection and removal of badly tracking tilts that would reduce FOV
- Automatic refinement of alignment to reduce residual error
- Parallel processing of multiple tilt series
- Auto-discovery of tilt series in data directories
- Optional updating of Warp XML and tomostar files
- Optional tomogram reconstruction for quick visualization

## Installation

```bash
git clone https://github.com/shahpnmlab/automateImod.git
conda create -n automateImod python=3.10 -y
conda activate automateImod
cd automateImod
pip install -e .
```

Verify that IMOD is installed in your user account and in your `$PATH`. Running `which 3dmod` should show the path to your `3dmod` executable.

Verify the package installation with:
```bash
automateImod --help
```

## Directory Structure

The expected directory structure for running automateImod is:

```
/path/to/IMOD/
└── TS_BASENAME/
    ├── TS_BASENAME.{mrc,st}
    └── TS_BASENAME.rawtlt

/path/to/MDOCS/
└── TS_BASENAME.{mrc.mdoc,mdoc}

/path/to/TOMOSTAR/
└── TS_BASENAME.tomostar

/path/to/XML/
└── TS_BASENAME.xml
```

## Usage

### Single Tilt Series Alignment

```bash
automateImod align-tilts \
  --ts-basename Position_1 \
  --ts-data-path /path/to/IMOD/ \
  --ts-mdoc-path /path/to/MDOCS/ \
  --ts-tilt-axis 84.7 \
  --ts-bin 6 \
  --ts-patch-size 300
```

### Parallel Processing Multiple Tilt Series

With the parallel processing option, automateImod will automatically discover all tilt series in the specified directories:

```bash
automateImod align-tilts \
  --ts-data-path /path/to/IMOD/ \
  --ts-mdoc-path /path/to/MDOCS/ \
  --ts-tilt-axis 84.7 \
  --ts-bin 6 \
  --ts-patch-size 300 \
  --n-cpu 8
```

### Complete Parallel Pipeline

Process all tilt series in a directory with all options enabled:

```bash
automateImod align-tilts \
  --ts-data-path /path/to/IMOD/ \
  --ts-mdoc-path /path/to/MDOCS/ \
  --ts-tomostar-path /path/to/TOMOSTAR/ \
  --ts-xml-path /path/to/XML/ \
  --ts-tilt-axis 84.7 \
  --ts-bin 6 \
  --ts-patch-size 300 \
  --min-fov 0.8 \
  --max-attempts 5 \
  --n-cpu 8 \
  --update-warp-xml \
  --reconstruct
```

### Legacy Commands

#### Update Warp XML Only

```bash
automateImod update-warp-xml \
  --ts-basename Position_1 \
  --ts-xml-path /path/to/XML/ \
  --ts-tomostar-path /path/to/TOMOSTAR/ \
  --ts-log-path /path/to/IMOD/
```

#### Reconstruct Tomograms Only

```bash
automateImod reconstruct-tomograms \
  --ts-data-path /path/to/IMOD/ \
  --ts-basename Position_1 \
  --ts-extension mrc \
  --tomo-bin 6
```

## Changelog

### v0.5.0
- Added parallel processing capabilities
- Auto-discovery of tilt series in data directories
- Integrated update-warp-xml and reconstruct-tomograms as command-line flags
- Improved logging and error handling
- Code refactoring for better modularity and readability

### v0.4.3
- Introduction of `--min_fov` arg for consistent removal of aberrant views
- Relaxed requirement for `.mdoc` files extension to include `.mrc.mdoc`
- Richer `TiltSeries` class

### v0.4.2
- Updates warp xml files by changing the \<frame\>.xml to \<frame\>.xml.bkp

### v0.4.1
- Introduced dark tilt detection routine
- Automatic detection of tilt series extension