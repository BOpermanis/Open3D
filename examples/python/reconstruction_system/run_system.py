# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/run_system.py

import json
import argparse
import time
import datetime
import sys
from os.path import isfile
from initialize_config import initialize_config
import open3d as o3d

sys.path.append("../utility")
from file import check_folder_structure
from os.path import join
from pprint import pprint
sys.path.append(".")

config_path = join("C:\\", "Users", "bruno", "repos", "IR3D", "rsconfig.json")
# print(config_path)
# exit()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("--config", default=config_path, help="path to the config file")
    parser.add_argument("--make", default=True,
                        help="Step 1) make fragments from RGBD sequence.",
                        action="store_true")
    parser.add_argument(
        "--register", default=True,
        help="Step 2) register all fragments to detect loop closure.",
        action="store_true")
    parser.add_argument("--refine", default=True,
                        help="Step 3) refine rough registrations",
                        action="store_true")
    parser.add_argument(
        "--integrate", default=True,
        help="Step 4) integrate the whole RGBD sequence to make final mesh.",
        action="store_true")
    parser.add_argument(
        "--slac", default=True,
        help="Step 5) (optional) run slac optimisation for fragments.",
        action="store_true")
    parser.add_argument(
        "--slac_integrate", default=False,
        help="Step 6) (optional) integrate fragements using slac to make final "
             "pointcloud / mesh.",
        action="store_true")
    parser.add_argument("--debug_mode",
                        help="turn on debug mode.",
                        action="store_true")
    parser.add_argument(
        '--device',
        help="(optional) select processing device for slac and slac_integrate. "
             "[example: cpu:0, cuda:0].",
        type=str,
        default='cpu:0')

    args = parser.parse_args()
    if not args.make and \
            not args.register and \
            not args.refine and \
            not args.integrate and \
            not args.slac and \
            not args.slac_integrate:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # check folder structure
    if True:  # args.config is not None:

        with open(args.config) as json_file:
            config = json.load(json_file)
            config["path_dataset"] = join("C:\\", "Users", "bruno", "data", "from_realsense", "test1", "realsense.bag")
            initialize_config(config)
            check_folder_structure(config["path_dataset"])
    assert config is not None

    # pprint(config)
    # exit()

    if args.debug_mode:
        config['debug_mode'] = True
    else:
        config['debug_mode'] = False

    config['device'] = args.device

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0, 0, 0, 0, 0, 0]
    if args.make:
        start_time = time.time()
        import make_fragments

        make_fragments.run(config)
        times[0] = time.time() - start_time
    if args.register:
        start_time = time.time()
        import register_fragments

        register_fragments.run(config)
        times[1] = time.time() - start_time
    if args.refine:
        start_time = time.time()
        import refine_registration

        refine_registration.run(config)
        times[2] = time.time() - start_time
    if args.integrate:
        start_time = time.time()
        import integrate_scene

        integrate_scene.run(config)
        times[3] = time.time() - start_time
    if args.slac:
        start_time = time.time()
        import slac

        config["method"] = "rigid"
        slac.run(config)
        times[4] = time.time() - start_time
    if args.slac_integrate:
        start_time = time.time()
        import slac_integrate

        slac_integrate.run(config)
        times[5] = time.time() - start_time

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- SLAC                %s" % datetime.timedelta(seconds=times[4]))
    print("- SLAC Integrate      %s" % datetime.timedelta(seconds=times[5]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()
