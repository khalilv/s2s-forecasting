# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from s2s.utils.datamodule import GlobalForecastDataModule
from s2s.climaX.module import GlobalForecastModule
from pytorch_lightning.cli import LightningCLI

# 1) entry point high-level class for training climaX. 

def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=GlobalForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_lat2d(cli.datamodule.normalize_data)
    cli.model.set_history(cli.datamodule.get_history())
    cli.model.set_hrs_each_step(cli.datamodule.get_hrs_each_step())
    cli.model.set_variables(cli.datamodule.in_variables, cli.datamodule.static_variables, cli.datamodule.out_variables)
    cli.model.set_plot_variables(cli.datamodule.plot_variables)
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    cli.model.init_metrics()
    cli.model.init_network()

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
