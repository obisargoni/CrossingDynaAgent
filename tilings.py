import numpy as np

class Tiling(object):
    """2D rectangular tiling.
    
    Arguments:
        limits {list} -- Min and max value tuple for each dimension.
        ntiles {iterable} -- Number of tiles for each dimension.
        offsets {iterable or None} -- Offset for each tile as multiple of tile width. No offset if None (default: None).
    """
    def __init__(self, limits, ntiles, offsets=None):
        self.ndim = len(limits)
        self.limits = limits
        self.ntiles = ntiles
        self.N = np.product(ntiles)
        self.offsets = offsets
        
        edges_and_widths = [np.linspace(self.limits[i][0], self.limits[i][1],
                                        self.ntiles[i]+1, retstep=True)
                            for i in range(self.ndim)]
        
        self.edges = [ew[0][1:-1] for ew in edges_and_widths]
        self.widths = [ew[1] for ew in edges_and_widths]
        
        if offsets is not None:
            for i in range(self.ndim):
                # Book Page 219: Offsets scaled relative to tile width
                self.edges[i] += self.offsets[i] * self.widths[i]
            
    def tile_dims(self, s):
        """Get tile index for each dimension for a given state s.
        
        Arguments:
            s {iterable} -- values representing the state.
        
        Returns:
            list -- tile index for each dimension
        """
        return [np.digitize(s[i], self.edges[i]) for i in range(self.ndim)]
        
    def tile(self, s):
        """Get index of tile activated by state s.
        
        Arguments:
            s {iterable} -- values representing the state
        
        Returns:
            int -- Tile index.
        """
        dims = self.tile_dims(s)
        tile = sum([dims[i] * np.product(self.ntiles[(i+1):])
                    for i in range(self.ndim-1)])
        tile = tile + dims[-1]
        
        return tile
    
    def feature(self, s):
        """Get feature vector for state s.
        
        Arguments:
            s {iterable} -- values representing the state
        
        Returns:
            np.Array -- Feature vector of length self.ntiles, all zeros except
            one for activated tile.
        """
        tile = self.tile(s)
        features = np.zeros(self.N)
        features[tile] = 1
        return features


class TilingGroup(object):
    """Set of Tiling objects with appropriate offsets between them.
    
    Arguments:
        ntilings {int} -- Number of tilings to generate. Book page 220 suggests at least 4 times number of dimensions.
        limits {list} -- Min and max value tuple for each dimension (same used for each tiling).
        ntiles {iterable} -- Number of tiles for each dimension (same used for each tiling).
    """
    def __init__(self, ntilings, limits, ntiles):
        self.ntilings = ntilings
        self.ntiles = ntiles
        self.limits = limits
        self.ndim = len(limits)
        
        # Book page 219: Offsets scaled by 1/ntilings.
        # Book page 220: Offsets tilings by (1, 3, 5...) units per dimension.
        self.offset_per_tiling = (1/self.ntilings)*np.arange(1, 2*self.ndim, 2)
        
        self.tilings = [Tiling(limits, ntiles,
                               offsets=i*self.offset_per_tiling)
                        for i in range(self.ntilings)]
        
        self.N = sum([t.N for t in self.tilings])
        
    def feature(self, s):
        if not isinstance(s, (list, tuple)):
            s = [s]
        features = np.array([t.feature(s) for t in self.tilings])
        return features.flatten()
