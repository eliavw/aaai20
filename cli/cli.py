"""
Entry point.

Command line script that;
    1. Makes a RunExp
    2. Extracts its commands
    3. Runs those commands through run_remote.
"""

import argparse
import json
import os
import subprocess
import sys
from os.path import abspath, dirname, relpath

import dill as pkl
import pandas as pd
from aaai20.io import build_filesystem

# Change with appropriate python paths
LOCAL_CONDA = "/home/elia/miniconda3/envs/aaai20/bin"
REMOTE_CONDA = "/cw/dtaijupiter/NoCsBack/dtai/elia/00_Software/anaconda/envs/aaai20/bin"

FS = build_filesystem(root=dirname(os.getcwd()))

def _build_environment():

    # Init
    env = {}

    # Define (hardcoding)
    cli_directory = FS['cli']
    exe_directory = FS['exe']

    env["commands_fname"] = os.path.join(cli_directory, "commands.csv")
    env["nodefile_fname"] = os.path.join(cli_directory, "nodefile")
    env["exe_atom_fname"] = os.path.join(exe_directory, "execute_atom.py")
    env["run_remote_fname"] = os.path.join(exe_directory, "run_remote")
    env["run_local_fname"] = os.path.join(exe_directory, "run_local")
    
    env["remote_conda"] = REMOTE_CONDA
    env["local_conda"] = LOCAL_CONDA

    return env

ENV = _build_environment()


def main(commands_file_or_folder, local=True):
    # Inputs
    commands_fnames = _build_command_fnames(commands_file_or_folder, scan_cli_config_folder=True)

    # Actions
    commands_df = _build_commands(commands_fnames)
    ENV["nb_procs"] = str(commands_df.shape[0] - 1)

    # Outputs
    commands_df.to_csv(ENV["commands_fname"])

    # Execution
    _execute_command_subprocess(local=local, **ENV)

    return


# Helpers
def _build_command_fnames(commands_file_or_folder, scan_cli_config_folder=False):
    """
    Build list of config files. If a file was provided, embed in a list.
    """

    if os.path.isfile(commands_file_or_folder):
        commands_fnames = [commands_file_or_folder]
    elif os.path.isdir(commands_file_or_folder):
        commands_fnames = os.listdir(commands_file_or_folder)
        commands_fnames = [os.path.join(commands_file_or_folder, f) for f in commands_fnames]
    elif scan_cli_config_folder:
        # We try to look in the cli-config folder.
        commands_file_or_folder = os.path.join(FS["cli-config"], commands_file_or_folder)
        commands_fnames = _build_command_fnames(commands_file_or_folder, scan_cli_config_folder=False)
    else:
        msg = """
        Input parameter is neither file nor folder: {}
        """.format(
            commands_file_or_folder
        )
        raise ValueError(msg)

    return commands_fnames


def _build_commands(fnames, shuffle=True):
    dfs = [pd.read_csv(f) for f in fnames]
    
    df = pd.DataFrame()
    df = pd.concat(dfs)

    if shuffle:
        # Shuffle all commands, cf. https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        df = df.sample(frac=1).reset_index(drop=True)
    return df


def _execute_command_subprocess(local=True, **kwargs):

    env = {**kwargs, **os.environ.copy()}

    # Remote or Local
    if local:
        bash = env["run_local_fname"]
    else:
        bash = env["run_remote_fname"]

    # Execute
    subprocess.call(bash, env=env)

    return


# CLI
def create_parser():
    """
    CLI Parser.

    Returns
    -------

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands", "-c", type=str, help="commands_file_or_folder_outer_scope")
    parser.add_argument(
        "--local", "-l", help="local_outer_scope, local yes/no", action="store_true"
    )

    return parser


# For executable script
if __name__ == "__main__":

    parser = create_parser()

    args = parser.parse_args()
    commands_file_or_folder_outer_scope = args.commands
    local_outer_scope = args.local

    main(commands_file_or_folder_outer_scope, local=local_outer_scope)
