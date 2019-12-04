import warnings

import dolphindb as ddb


_ddb_session = ddb.session()


def default_session():
    return _ddb_session


_orca_verbose = False


def _set_verbose(verbose):
    global _orca_verbose
    _orca_verbose = verbose


def _get_verbose():
    return _orca_verbose


class CopiedTableWarning(Warning):
    pass


class AttachDefaultIndexWarning(Warning):
    pass


class NotDolphinDBIdentifierWarning(Warning):
    pass


class MixedTypeWarning(Warning):
    pass


class ApplyCallableWarning(Warning):
    pass


def warn_not_dolphindb_identifier():
    warnings.warn("The DataFrame contains an invalid column name for "
                  "DolphinDB. Will convert to an automatically "
                  "generated column name.", NotDolphinDBIdentifierWarning)

def warn_apply_callable():
    warnings.warn("Applying a callable to an Orca object will converting "
                  "the object to pandas object, applying the callable, "
                  "and converting back to an Orca object", ApplyCallableWarning)
