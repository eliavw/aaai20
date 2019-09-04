import os
from os.path import dirname


def original_filename(basename, extension="arff"):

    root_dir = dirname(os.getcwd())
    true_path = os.path.join(
        root_dir,
        "data",
        "raw",
        "datasets-UCI",
        "UCI",
        "{}.{}".format(basename, extension),
    )

    return os.path.relpath(true_path)


def filename_dataset(
    basename, step=1, prefix="", suffix="", separator="-", extension="arff", check=True
):
    """
    Filename generator for the datafiles of this experiment
    """

    filename = build_filename(
        basename, prefix=prefix, suffix=suffix, separator=separator, extension=extension
    )

    # FS things
    root_dir = dirname(os.getcwd())
    data_dir = os.path.relpath(os.path.join(root_dir, "data"))
    step_dir = os.path.join(data_dir, "step-" + str(step).zfill(2))

    # If dir does not exist, make it
    if check:
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)

    return os.path.join(step_dir, filename)


def filename_query(
    basename, prefix="", suffix="", separator="-", extension="npy", check=True
):
    """
    Filename generator of the query files of this experiment
    """
    filename = build_filename(
        basename, prefix=prefix, suffix=suffix, separator=separator, extension=extension
    )

    # FS things
    root_dir = dirname(os.getcwd())
    conf_dir = os.path.relpath(os.path.join(root_dir, "config"))
    qry_dir = os.path.join(conf_dir, "query")

    # If dir does not exist, make it
    if check:
        if not os.path.exists(qry_dir):
            os.makedirs(qry_dir)

    return os.path.join(qry_dir, filename)


def filename_results(basename, prefix="", suffix="", separator="-", extension="csv", check=True):
    filename = build_filename(
        basename, prefix=prefix, suffix=suffix, separator=separator, extension=extension
    )

    # FS things
    root_dir = dirname(os.getcwd())
    results_dir = os.path.relpath(os.path.join(root_dir, "results"))

    # If dir does not exist, make it
    if check:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
    return os.path.join(results_dir, filename)


def build_filename(basename, prefix="", suffix="", separator="-", extension="csv"):
    return separator.join(
        [x for x in (prefix, basename, suffix) if len(x) > 0]
    ) + ".{}".format(extension)
