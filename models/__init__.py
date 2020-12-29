# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args, strategy):
    return build(args, strategy)
