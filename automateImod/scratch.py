import pio
"""
--ts-data-path /Users/ps/data/wip/automateImod/example_data/Frames --ts-mdoc-path  /Users/ps/data/wip/automateImod/example_data/mdoc
--ts-xml-path /Users/ps/data/wip/automateImod/example_data/Frames --ts-basename lam1_poly1_ts_006 --ts-extension st --ts-tilt-axis 84.7
--ts-bin 2 --ts-patch-size 250
"""
path_to_data = "/Users/ps/data/wip/automateImod/example_data/Frames"
path_to_xml_data = "/Users/ps/data/wip/automateImod/example_data/Frames"
path_to_mdoc = "/Users/ps/data/wip/automateImod/example_data/mdoc"
basename = "lam1_poly1_ts_006"
extension = "st"
tilt_axis_ang = "84.7"
ts_bin = "2"
ts_patch_size = "256"


ts = pio.TiltSeries(path_to_ts_data=path_to_data, path_to_xml_data=path_to_xml_data, path_to_mdoc_data=path_to_mdoc,
                    basename=basename, extension=extension,
                    tilt_axis_ang=tilt_axis_ang, binval=ts_bin, patch_size=ts_patch_size)
print(ts)