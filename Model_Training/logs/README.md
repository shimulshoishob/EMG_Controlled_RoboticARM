<div align="center">

# 🗒️ `Model_Training/logs/`

**Training logs**

[↑ Model_Training](../README.md)

</div>

`training_<run_id>.log`: one file per Training Studio run (`run_id` is a timestamp such as `20261124_143936`): the settings used, per-model progress, durations and any error with its traceback. A failing model is logged here and the run continues.

The CLI benchmark writes its own `training.log` inside its `results/experiment_NNN/` folder.
