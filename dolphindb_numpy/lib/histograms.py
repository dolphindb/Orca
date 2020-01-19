from dolphindb_numpy.core.common import _unary_op, _binary_op


def histogram(a, bins=10, range=None, normed=None, weights=None,
              density=None):
    return _unary_op("histogram", orca_support=False)(a, range=range, normed=normed, weights=weights,
                                                      density=density)

def histogram2d(x, y, bins=10, range=None, normed=None, weights=None,
                density=None):
    return _binary_op("histogram2d", orca_support=False)(x, y, range=range, normed=normed, weights=weights,
                                                      density=density)

def histogramdd(sample, bins=10, range=None, normed=None, weights=None,
                density=None):
    return _unary_op("histogramdd", orca_support=False)(sample, range=range, normed=normed, weights=weights,
                                                      density=density)

def histogram_bin_edges(a, bins=10, range=None, weights=None):
    return _unary_op("histogram_bin_edges", orca_support=False)(a, range=range, weights=weights)