import click
from .config import Config
import logging

pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    "--skeleton-csv",
    default=None,
    type=click.Path(),
    help="A csv file with data in (nid,pid,x,y,z) format",
    required=True,
)
@click.option(
    "--sarbor-config",
    default=None,
    type=click.Path(),
    help="Configuration file for sarbor",
    required=True,
)
@click.option(
    "--output-file",
    default="-",
    type=click.Path(),
    help="output file base. Directory in which to store your data.",
)
@pass_config
def cli(
    segmentation_source: str,
    skeleton_csv: click.Path,
    sarbor_config: click.Path,
    output_file: click.Path,
):
    config = Config.from_toml(sarbor_config)
    config.skeleton.output_file_base = output_file
    config.skeleton.skeleton_csv = skeleton_csv


@cli.command()
@pass_config
def watershed(config):
    from sarbor import query_watershed

    query_watershed(config)


@cli.command()
@click.option(
    "--model-weights-file",
    default=None,
    type=click.Path(),
    help="Model weights file for NN based segmentation sources (i.e. Diluvian)",
)
@click.option(
    "--model-training-config",
    default=None,
    type=click.Path(),
    help=(
        "Model config for NN based segmentation sources (i.e. Diluvian). "
        + "This should be the configuration used for training"
    ),
)
@click.option(
    "--model-job-config",
    default=None,
    type=click.Path(),
    help=(
        "Model config for NN based segmentation sources (i.e. Diluvian). "
        + "This file makes job specific changes to diluvian config"
    ),
)
@click.option(
    "--volume-file",
    default=None,
    type=click.Path(),
    help="Volume config file for the raw image data for NN based approaches (i.e. Diluvian)",
)
@pass_config
def diluvian(config):
    logging.info("diluvian")
