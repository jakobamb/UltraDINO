# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import sys

from dotenv import load_dotenv

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser, main as train_main


def main():
    load_dotenv()

    parser = get_args_parser()
    args = parser.parse_args()

    setup_logging()
    train_main(args)


if __name__ == "__main__":
    sys.exit(main())
