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


def _raise_must_compute_error(msg):
    raise ValueError(msg + ", use .compute() to explicitly convert "
                     "the Expression to a DataFrame or Series")


def _warn_not_dolphindb_identifier():
    warnings.warn("The DataFrame contains an invalid column name for "
                  "DolphinDB. It will be converted to an automatically "
                  "generated column name.", NotDolphinDBIdentifierWarning)


def _warn_apply_callable():
    warnings.warn("Applying a callable to an Orca object will convert "
                  "the object to a pandas object, apply the callable, "
                  "and then convert back to an Orca object", ApplyCallableWarning)


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

