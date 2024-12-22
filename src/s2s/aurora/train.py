# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from s2s.utils.datamodule import GlobalForecastDataModule
from s2s.aurora.module import GlobalForecastModule
from pytorch_lightning.cli import LightningCLI

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
    assert cli.datamodule.in_variables == cli.datamodule.out_variables, "Input and output variables must be the same for Aurora"
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_variables(cli.datamodule.in_variables, cli.datamodule.static_variables, cli.datamodule.out_variables)
    cli.model.set_plot_variables(cli.datamodule.plot_variables)
    if cli.datamodule.normalize_data:
        cli.model.set_denormalization(cli.datamodule.get_transforms('out'))
    if cli.model.use_default_statistics:    
        in_mean, in_std = cli.model.get_default_aurora_normalization_stats(cli.datamodule.in_variables)
        out_mean, out_std = cli.model.get_default_aurora_normalization_stats(cli.datamodule.out_variables)
        static_mean, static_std = cli.model.get_default_aurora_normalization_stats(cli.datamodule.static_variables)
        cli.datamodule.update_normalization_stats(in_mean, in_std, 'in')
        cli.datamodule.update_normalization_stats(out_mean, out_std, 'out')
        cli.datamodule.update_normalization_stats(static_mean, static_std, 'static')
    cli.model.init_metrics()
    cli.model.init_network()

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
