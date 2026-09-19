"""Mark incomplete tasks from a failed scheduler launch interrupted."""

import argparse

from timebench.pipeline.runs import interrupt_launch

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("--launch-id", required=True)
args = parser.parse_args()
interrupt_launch(args.root, args.launch_id)
