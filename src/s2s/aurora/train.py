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
    assert cli.datamodule.normalize_data is False, "normalize_data must be false for Aurora. The model handles normalization internally."
    assert cli.datamodule.in_variables == cli.datamodule.out_variables, "Input and output variables must be the same for Aurora"
    cli.model.update_normalization_stats(cli.datamodule.in_variables, *cli.datamodule.get_normalization_stats(cli.datamodule.in_variables))
    cli.model.update_normalization_stats(cli.datamodule.static_variables, *cli.datamodule.get_normalization_stats(cli.datamodule.static_variables, "static"))
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_variables(cli.datamodule.in_variables, cli.datamodule.static_variables, cli.datamodule.out_variables)
    cli.model.set_val_clim(cli.datamodule.val_clim, cli.datamodule.val_clim_timestamps)
    cli.model.set_test_clim(cli.datamodule.test_clim, cli.datamodule.val_clim_timestamps)
    cli.model.set_delta_time(cli.datamodule.predict_step_size, cli.datamodule.hrs_each_step)
    cli.model.init_metrics()
    cli.model.init_network()

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
