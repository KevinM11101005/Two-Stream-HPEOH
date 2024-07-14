# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

from mmengine.structures import InstanceData
from torch import Tensor

from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import InstanceList

def decode(decoder, batch_outputs: Union[Tensor,
                                        Tuple[Tensor]]) -> InstanceList:
    """Decode keypoints from outputs.

    Args:
        batch_outputs (Tensor | Tuple[Tensor]): The network outputs of
            a data batch

    Returns:
        List[InstanceData]: A list of InstanceData, each contains the
        decoded pose information of the instances of one data sample.
    """

    def _pack_and_call(args, func):
        if not isinstance(args, tuple):
            args = (args, )
        return func(*args)

    if decoder is None:
        raise RuntimeError(
            f'Please set the decoder configs in the init parameters to '
            'enable head methods `head.predict()` and `head.decode()`')

    if decoder.support_batch_decoding:
        batch_keypoints, batch_scores = _pack_and_call(
            batch_outputs, decoder.batch_decode)
        if isinstance(batch_scores, tuple) and len(batch_scores) == 2:
            batch_scores, batch_visibility = batch_scores
        else:
            batch_visibility = [None] * len(batch_keypoints)

    else:
        batch_output_np = to_numpy(batch_outputs, unzip=True)
        batch_keypoints = []
        batch_scores = []
        batch_visibility = []
        for outputs in batch_output_np:
            keypoints, scores = _pack_and_call(outputs,
                                                decoder.decode)
            batch_keypoints.append(keypoints)
            if isinstance(scores, tuple) and len(scores) == 2:
                batch_scores.append(scores[0])
                batch_visibility.append(scores[1])
            else:
                batch_scores.append(scores)
                batch_visibility.append(None)

    preds = []
    for keypoints, scores, visibility in zip(batch_keypoints, batch_scores,
                                                batch_visibility):
        pred = InstanceData(keypoints=keypoints, keypoint_scores=scores)
        if visibility is not None:
            pred.keypoints_visible = visibility
        preds.append(pred)

    return preds