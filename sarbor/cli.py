import click
from .config import Config
import logging


@click.command()
@click.option(
    "--segmentation_source",
    default=None,
    type=str,
    help="Where to obtain segmentations. Currently supported sources are diluvian and watershed",
    required=True,
)
@click.option(
    "--skeleton_csv",
    default=None,
    type=click.Path(),
    help="A csv file with data in (nid,pid,x,y,z) format",
    required=True,
)
@click.option(
    "--sarbor_config",
    default=None,
    type=click.Path(),
    help="Configuration file for sarbor",
    required=True,
)
@click.option(
    "--model_weights_file",
    default=None,
    type=click.Path(),
    help="Model weights file for NN based segmentation sources (i.e. Diluvian)",
)
@click.option(
    "--model_training_config",
    default=None,
    type=click.Path(),
    help=(
        "Model config for NN based segmentation sources (i.e. Diluvian). "
        + "This should be the configuration used for training"
    ),
)
@click.option(
    "--model_job_config",
    default=None,
    type=click.Path(),
    help=(
        "Model config for NN based segmentation sources (i.e. Diluvian). "
        + "This file makes job specific changes to diluvian config"
    ),
)
@click.option(
    "--volume_file",
    default=None,
    type=click.Path(),
    help="Volume config file for the raw image data for NN based approaches (i.e. Diluvian)",
)
@click.option(
    "--output_file",
    default="-",
    type=click.Path(),
    help="output file base. Directory in which to store your data.",
)
def cli(
    segmentation_source,
    skeleton_csv: click.Path,
    sarbor_config: click.Path,
    model_weights_file: click.Path,
    model_training_config: click.Path,
    model_job_config: click.Path,
    volume_file: click.Path,
    output_file: click.Path,
):
    config = Config()
    logging.info(config)
