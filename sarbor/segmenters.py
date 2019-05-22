from skeletons import Skeleton


class Segmenter:
    """
    A source for querying segmentations
    """

    def __init__(self, **kwargs):
        self.stuff = kwargs

    def segment_skeleton(self, skel: Skeleton):
        """
        Takes a skeleton and segments around each node
        masks returned do not need to be full resolution,
        but do need to be a factor of the desired resolution
        i.e. if desired resolution is 10x10x10 nm, return 2x5x5
        is fine because you can simply average the pixels
        """
        raise NotImplementedError

    def create_queues(self):
        """
        create queues for handling nodes
        """
        raise NotImplementedError

    def process_result(self):
        """
        do any necessary processing
        """
        raise NotImplementedError

    def queue_next(self):
        """
        queue the next node
        """
        raise NotImplementedError
