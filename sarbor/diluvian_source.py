import logging
import numpy as np
import os

from .skeletons import Skeleton


from collections import deque
from multiprocessing import Manager, Process

from tqdm import tqdm

from diluvian.config import CONFIG
from diluvian.volumes import SubvolumeBounds
from diluvian.regions import Region

logger = logging.getLogger('sarbor')


class DiluvianSource:
    def __init__(self, volumes, config, model_file):
        self.source = "diluvian"
        self.volume = volumes[list(volumes.keys())[0]]
        self.config = config
        self.model_file = model_file

    def worker(
        self,
        worker_id,
        set_devices,
        volume,
        region_shape,
        nodes,
        results,
        lock,
        revoked,
    ):
        lock.acquire()
        import tensorflow as tf

        if set_devices:
            # Only make one GPU visible to Tensorflow so that it does not allocate
            # all available memory on all devices.
            # See: https://stackoverflow.com/questions/37893755
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)

        with tf.device("/gpu:0"):
            # Late import to avoid Keras import until TF bindings are set.
            from diluvian.network import load_model

            logger.debug("Worker %s: loading model", worker_id)
            model = load_model(self.model_file, CONFIG.network)
        lock.release()

        def is_revoked(test_node):
            ret = False
            lock.acquire()
            if tuple(test_node) in revoked:
                ret = True
                revoked.remove(tuple(test_node))
            lock.release()
            return ret

        while True:
            node = nodes.get(True)

            if not isinstance(node, np.ndarray):
                logger.debug("Worker %s: got DONE", worker_id)
                break

            if is_revoked(node):
                results.put((node, None))
                continue

            def stopping_callback(region):
                stop = is_revoked(node)
                if (
                    self.reject_non_seed_components
                    and region.bias_against_merge
                    and region.mask[tuple(region.seed_vox)] < 0.5
                ):
                    stop = True
                return stop

            logger.debug("Worker %s: got seed %s", worker_id, str(node))

            logger.debug(
                "start: {0}".format(node[2:] - np.floor_divide(region_shape, 2))
            )
            logger.debug(
                "stop: {0}".format(node[2:] + np.floor_divide(region_shape, 2) + 1)
            )

            image = volume.get_subvolume(
                SubvolumeBounds(
                    start=(node[:]) - np.floor_divide(region_shape, 2),
                    stop=(node[:]) + np.floor_divide(region_shape, 2) + 1,
                )
            ).image

            # Flood-fill and get resulting mask.
            # Allow reading outside the image volume bounds to allow segmentation
            # to fill all the way to the boundary.
            region = Region(
                image,
                seed_vox=np.floor_divide(np.array(image.shape), 2) + 1,
                sparse_mask=False,
                block_padding="reflect",
            )
            region.bias_against_merge = self.bias
            try:
                next(
                    region.fill(
                        model,
                        move_batch_size=self.move_batch_size,
                        max_moves=self.max_moves,
                        stopping_callback=stopping_callback,
                        remask_interval=self.remask_interval,
                    )
                )
            except Region.EarlyFillTermination:
                logger.debug("Worker %s: node %s failed to fill", worker_id, str(node))
            except StopIteration:
                pass
            logger.debug("Worker %s: node %s filled", worker_id, str(node))

            results.put((node, region.to_body()))

    def segment_skeleton(self, skel: Skeleton):
        region_shape = skel.seg.fov_shape_phys[::-1] // CONFIG.volume.resolution
        try:
            assert all(
                region_shape * CONFIG.volume.resolution == skel.seg.fov_shape_phys[::-1]
            ), "Incompatible field of view"
        except AssertionError:
            # This can happen because the model may be trained on some downsampled resolution i.e. 40x16x16 instead of 40x8x8.
            # This means that the output might not be simply convertable between the new resolution, and the desired resolution,
            # for example 40x40x40.
            # The simplest change to make would simply be to downsample the output of sarbor less, so that will be the default behavior.
            # This should be handled by the user interface to make sure that this sort of problem does not arise in the first place
            skel._config.segmentations.downsample_factor = (
                CONFIG.volume.resolution[::-1]
                // skel._config.segmentations.resolution_phys
            )
            skel._config.segmentations.fov_shape_voxels = (
                skel._config.segmentations.fov_shape_phys
                // skel._config.segmentations.voxel_resolution
            )
            skel._config.segmentations.fov_shape_voxels += (
                skel._config.segmentations.fov_shape_voxels + 1
            ) % 2
            skel._config.segmentations.fov_shape_phys = (
                skel._config.segmentations.fov_shape_voxels
                * skel._config.segmentations.voxel_resolution
            )
            skel._config.segmentations.validate_fov_shape()
            logger.warn(
                "New fov shape: {}".format(skel._config.segmentations.fov_shape_phys)
            )
        return {
            key: value.transpose([2, 1, 0])
            for key, value in self.fill_skeleton_with_model_threaded_new(
                {node.key: node.value.center[::-1] for node in skel.get_nodes()},
                num_workers=8,
                region_shape=skel.seg.fov_shape_phys[::-1],
            ).items()
        }

    def fill_skeleton_with_model_threaded_new(
        self,
        nodes,
        partition=False,
        augment=False,
        bounds_input_file=None,
        bias=True,
        move_batch_size=1,
        max_moves=None,
        remask_interval=None,
        sparse=False,
        moves=None,
        num_workers=1,
        worker_prequeue=1,
        reject_early_termination=False,
        reject_non_seed_components=True,
        region_shape=CONFIG.model.input_fov_shape,
    ):
        """
        Floodfill small regions around a list of seed points.
        Necessary inputs:
        - model_file
        - seeds
        - volume
        - config

        returns Dict[seed: Mask]
        """

        self.bias = bias
        self.move_batch_size = move_batch_size
        self.max_moves = max_moves
        self.remask_interval = remask_interval
        self.reject_non_seed_components = reject_non_seed_components

        # Get Volume
        self.volume = self.volume.downsample(CONFIG.volume.resolution)

        # Seeds come in real coordinates
        node_ids, seeds = zip(
            *[
                (
                    nid,
                    np.array(
                        self.volume.world_coord_to_local(
                            self.volume.real_coord_to_world(seed)
                        )
                    ),
                )
                for nid, seed in nodes.items()
            ]
        )
        map_back = {tuple(seed): nid for seed, nid in zip(seeds, node_ids)}
        logger.warning(
            "CONFIG input_fov_shape: {} vs Sarbor fov_shape: {}".format(
                CONFIG.model.input_fov_shape,
                (
                    region_shape,
                    CONFIG.volume.resolution,
                    region_shape / CONFIG.volume.resolution,
                ),
            )
        )
        region_shape = region_shape // CONFIG.volume.resolution

        pbar = tqdm(desc="Seed queue", total=len(seeds), miniters=1, smoothing=0.0)
        num_nodes = len(seeds)
        seed_generator = iter(seeds)

        manager = Manager()
        # Queue of seeds to be picked up by workers.
        seed_queue = manager.Queue()
        # Queue of results from workers.
        results_queue = manager.Queue()
        # Dequeue of seeds that were put in seed_queue but have not yet been
        # combined by the main process.
        dispatched_seeds = deque()
        # Seeds that were placed in seed_queue but subsequently covered by other
        # results before their results have been processed. This allows workers to
        # abort working on these seeds by checking this list.
        revoked_seeds = manager.list()
        # Results that have been received by the main process but have not yet
        # been combined because they were not received in the dispatch order.
        unordered_results = {}

        final_results = {}

        def queue_next_seed():
            total = 0
            for seed in seed_generator:
                if unordered_results.get(tuple(seed)) is not None:
                    # This seed has already been filled.
                    total += 1
                    continue
                dispatched_seeds.append(seed)
                seed_queue.put(seed)

                break

            return total

        for _ in range(min(num_nodes, num_workers * worker_prequeue)):
            processed_nodes = queue_next_seed()
            pbar.update(processed_nodes)

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            set_devices = False
            num_workers = 1
            logger.warn(
                "Environment variable CUDA_VISIBLE_DEVICES is set, "
                + "so only one worker can be used.\n"
                + "See https://github.com/aschampion/diluvian/issues/11"
            )
        else:
            set_devices = True

        workers = []
        loading_lock = manager.Lock()
        for worker_id in range(num_workers):
            w = Process(
                target=self.worker,
                args=(
                    worker_id,
                    set_devices,
                    self.volume,
                    region_shape,
                    seed_queue,
                    results_queue,
                    loading_lock,
                    revoked_seeds,
                ),
            )
            w.start()
            workers.append(w)

        while dispatched_seeds:
            processed_seeds = 1
            expected_seed = dispatched_seeds.popleft()
            logger.debug("Expecting seed %s", expected_seed)

            if tuple(expected_seed) in unordered_results:
                logger.debug("Expected seed %s is in old results", expected_seed)
                seed = expected_seed
                body = unordered_results[tuple(seed)]
                del unordered_results[tuple(seed)]  # WHY

            else:
                seed, body = results_queue.get(True)
                processed_seeds += queue_next_seed()

                while not np.array_equal(seed, expected_seed):
                    logger.debug("Node %s is early, stashing", seed)
                    unordered_results[tuple(seed)] = body
                    seed, body = results_queue.get(True)
                    processed_seeds += queue_next_seed()

            logger.debug("Processing node at %s", seed)
            pbar.update(processed_seeds)

            if final_results.get(tuple(seed)) is not None:
                # This seed has already been filled.
                logger.debug(
                    "Seed (%s) was filled but has been covered in the meantime.",
                    np.array_str(seed),
                )
                loading_lock.acquire()
                if tuple(seed) in revoked_seeds:
                    revoked_seeds.remove(tuple(seed))
                loading_lock.release()
                continue

            if body is None:
                logger.warning("Body of Seed ({}) is None".format(seed))

                continue
                # REDO THIS SEED
                raise NotImplementedError
                continue

            if not body.is_seed_in_mask():
                logger.warning("Seed ({}) is not in its body.".format(seed))

                continue
                # REDO THIS SEED
                raise NotImplementedError
                continue

            mask, bounds = body._get_bounded_mask(CONFIG.postprocessing.closing_shape)

            body_size = np.count_nonzero(mask)

            if body_size == 0:
                logger.warning("Body of seed {} is empty.".format(seed))

                continue
                # REDO THIS SEED
                raise NotImplementedError
                continue

            final_results[map_back[tuple(seed)]] = mask
            logger.warning("Filled seed ({})".format(seed))

        for _ in range(num_workers):
            seed_queue.put("DONE")
        for wid, worker in enumerate(workers):
            worker.join()
        manager.shutdown()

        pbar.close()

        return final_results
