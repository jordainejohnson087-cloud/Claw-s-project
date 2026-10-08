#!/usr/bin/env python3
"""Console input/output for meter_detector.

  python console_io.py read IMAGE [--config cfg.json] [--json]
  python console_io.py set key=value [key=value ...] [--config cfg.json]
  python console_io.py show [--config cfg.json]
"""
import argparse
import json
import sys

import cv2

import meter_detector as md


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["read", "set", "show"])
    p.add_argument("args", nargs="*")
    p.add_argument("--config", default="meter_config.json")
    p.add_argument("--json", action="store_true", help="JSON output")
    a = p.parse_args(argv)
    try:
        if a.command == "show":
            print(json.dumps(md.load_config(a.config), indent=2))
        elif a.command == "set":
            updates = dict(item.split("=", 1) for item in a.args)
            print(json.dumps(md.update_config(a.config, updates), indent=2))
        else:
            if len(a.args) != 1:
                p.error("read requires exactly one IMAGE")
            img = cv2.imread(a.args[0])
            if img is None:
                raise ValueError(f"cannot read image: {a.args[0]}")
            cfg = md.load_config(a.config)
            res = md.detect(img, cfg)
            if a.json:
                print(json.dumps(res))
            else:
                print(f"{res['value']:.2f} {cfg['units']}".strip())
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
