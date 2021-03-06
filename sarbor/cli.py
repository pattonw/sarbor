import click
from .config import Config
import logging
import sys

logger = logging.getLogger("sarbor")
logger.setLevel(logging.INFO)
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(name)s: (%(levelname)s) %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

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
@click.option("--log-level", default=logging.INFO, type=int, help="Log level")
@pass_config
def cli(
    config: Config,
    skeleton_csv: click.Path,
    sarbor_config: click.Path,
    output_file: click.Path,
    log_level: int,
):
    console.setLevel(level=log_level)
    logger.setLevel(level=log_level)
    config.from_toml(sarbor_config)
    config.skeleton.output_file_base = output_file
    config.skeleton.csv = skeleton_csv


@cli.command()
@click.option(
    "--cached-lsd-config",
    default=None,
    type=click.Path(),
    help="Configuration file for cached lsd segmentation source.",
    required=True,
)
@pass_config
def cached_lsd(config, cached_lsd_config):
    from .sarbor import query_cached_lsd

    logger.warning(
        "Starting a cached lsd job with config: {}".format(cached_lsd_config)
    )

    query_cached_lsd(config, cached_lsd_config)


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
def diluvian(
    config, model_weights_file, model_training_config, model_job_config, volume_file
):
    from .sarbor import query_diluvian

    query_diluvian(
        config, model_weights_file, model_training_config, model_job_config, volume_file
    )
