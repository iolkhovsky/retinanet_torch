import numpy as np


class AnchorGenerator:

    def __init__(self, aspect_ratios=[0.5, 1., 2.], scales=[2 ** x for x in [0, 1./3., 2./3.]]):
        self.aspect_ratios = aspect_ratios
        self.scales = scales

    def _generate_base_ahcnors(self, base_size):
        num_anchors = len(self.aspect_ratios) * len(self.scales)
        anchors = np.zeros((num_anchors, 4))
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.aspect_ratios))).T
        areas = anchors[:, 2] * anchors[:, 3]
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.aspect_ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.aspect_ratios, len(self.scales))
        x_offsets, y_offsets = -0.5 * anchors[:, 2], -0.5 * anchors[:, 3]
        anchors[:, 0] += x_offsets
        anchors[:, 1] += y_offsets
        return anchors

    def generate(self, ref_img_sz, feat_map_sz):
        base_size = max(1, int(2 ** (np.log(np.ceil(ref_img_sz / feat_map_sz)) + 2)))
        base_anchors = self._generate_base_ahcnors(base_size)
        stride = ref_img_sz / feat_map_sz
        x = np.linspace(0.5, feat_map_sz - 0.5, feat_map_sz) * stride
        y = np.linspace(0.5, feat_map_sz - 0.5, feat_map_sz) * stride
        x_offsets, y_offsets = np.meshgrid(x, y)

        anchors = np.asarray([base_anchors] * feat_map_sz * feat_map_sz)
        anchors = np.reshape(anchors, newshape=(feat_map_sz, feat_map_sz, len(base_anchors), 4))
        for i in range(len(base_anchors)):
            anchors[:, :, i, 0] += x_offsets
            anchors[:, :, i, 1] += y_offsets

        return np.reshape(anchors, newshape=(-1, 4))


