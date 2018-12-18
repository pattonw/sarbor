class Region:
    """
    The region class contains information and operations specific to
    the neuron skeleton nodes.
    TODO: Think of different names to make this clearer
    """

    def __init__(self, center=None, mask=None):
        """
        bounds: [np.array, np.array]
        mask: np.array
        """
        self._center = center
        self._mask = mask
        self._center_of_mass = None

    @property
    def center(self):
        """
        Get the center of a region.
        Throws error if center is None
        TODO: Change constructor so that center cannot be None
        """
        if self._center is None:
            raise Exception("No center available")
        else:
            return self._center.astype(int)

    @center.setter
    def center(self, center):
        if self._center is None:
            self._center = center
        else:
            raise Exception("Overriding the center is not supported")

    @property
    def mask(self):
        if self._mask is None:
            # mask can be None
            return None
        else:
            return self._mask

    @mask.setter
    def mask(self, mask):
        if self._mask is None:
            self._mask = mask
        else:
            raise Exception("Overriding the mask is not supported")

    def get_bounds(self, fov_shape):
        return (self.center - fov_shape // 2, self.center + fov_shape // 2 + 1)

    def insert_values(self, values):
        if values is not None:
            fov_shape = values.get("fov_shape", None)
            if fov_shape is not None:
                self.fov_shape = fov_shape
            center = values.get("center", None)
            if center is not None:
                self.center = center
            mask = values.get("mask", None)
            if mask is not None:
                self.mask = mask

    def transform(self, translation, scale):
        """
        The only way to change center and field of view.
        This means the coordinate system of center and
        fov_shape will always match as long as they are
        """
        self._center = (self._center + translation) * scale

    def clone(self):
        new_region = type(self)(center=self._center.clone(), mask=self._mask.clone())
        return new_region
