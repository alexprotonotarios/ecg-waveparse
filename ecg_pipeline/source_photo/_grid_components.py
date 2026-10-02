"""Retain original-dark pixels linked through the existing contrast support mask."""
import cv2
import numpy as np

def anchored_dark_components(original_dark, colour, contrast_support):
    original = np.asarray(original_dark) > 0.5
    coloured = np.asarray(colour) > 0.5
    assert original.ndim == 2 and original.shape == coloured.shape
    support = np.asarray(contrast_support) > 0.5
    assert support.shape == original.shape and np.all(original <= support)
    count, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    anchors = original & ~coloured
    retained = np.zeros(count, dtype=bool)
    retained[np.unique(labels[anchors])] = True
    retained[0] = False
    result = (original & retained[labels]).astype(np.float32)
    assert np.all(result <= original)
    sizes = np.bincount(labels.ravel(), minlength=count)
    anchor_counts = np.bincount(labels[anchors], minlength=count)
    receipt = {'connectivity': 8, 'componentCount': int(count - 1), 'retainedComponentCount': int(retained.sum()), 'removedComponentCount': int(count - 1 - retained.sum()), 'supportPixels': int(support.sum()), 'originalDarkPixels': int(original.sum()), 'neutralAnchorPixels': int(anchors.sum()), 'retainedDarkPixels': int(result.sum()), 'removedDarkPixels': int(original.sum() - result.sum()), 'components': [{'label': i, 'pixels': int(sizes[i]), 'neutralAnchorPixels': int(anchor_counts[i]), 'retained': bool(retained[i])} for i in range(1, count)]}
    return (result, receipt)
