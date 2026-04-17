"""BioBB-style Optuna hyperparameter tuning for MDAE training."""
from __future__ import annotations

import os
import shutil
from typing import Any, Dict

from biobb_common.tools import file_utils as fu
from biobb_common.tools.file_utils import launchlogger
from biobb_common.generic.biobb_object import BiobbObject

from biobb_pytorch.mdae.hparam_tuning.objective import make_objective
from biobb_pytorch.mdae.hparam_tuning.merge_props import validate_search_space
from biobb_pytorch.mdae.hparam_tuning.study_factory import create_study_from_config
from biobb_pytorch.mdae.utils.log_utils import get_size


class HparamTuning(BiobbObject):
    """
    | biobb_pytorch HparamTuning
    | Hyperparameter search with Optuna over ``TrainModel`` properties (file I/O only).

    Args:
        input_model_pth_path (str): Path to the initial model ``.pth`` (same as ``train_model`` input).
        input_dataset_pt_path (str): Path to the dataset ``.pt`` (same as ``train_model`` input).
        output_model_pth_path (str): Path to write the best trial's full trained model (``torch.save``).
        properties (dict):
            * **Trainer** / **Dataset** — base config merged with suggested values (same as ``TrainModel``).
            * **Optuna** — tuning block:

              * **n_trials** (*int*) — number of Optuna trials.
              * **direction** (*str*) — ``minimize`` or ``maximize`` (default ``minimize``).
              * **sampler** (*dict*, optional) — e.g. ``{"type": "TPESampler", "seed": 42}``.
              * **pruner** (*dict*, optional) — e.g. ``{"type": "MedianPruner", "n_startup_trials": 5}``.
              * **storage** (*str*, optional) — RDB URL for persistent studies.
              * **study_name** (*str*, optional) — used with ``storage`` (default ``biobb_mdae_hparam``).
              * **load_if_exists** (*bool*, optional) — default ``True`` when ``storage`` is set.
              * **objective** (*dict*) — ``{"metric": "train_loss", "aggregation": "last"}``;
                ``aggregation``: ``last``, ``min``, ``mean``, ``max``.
              * **search_space** (*list*) — entries ``{"param": "Trainer.max_epochs", "type": "int",
                "low": 1, "high": 20}``, or ``float`` / ``categorical`` (see module docs).

    Examples:
        Minimal ``properties['Optuna']``::

            'Optuna': {
                'n_trials': 4,
                'objective': {'metric': 'train_loss', 'aggregation': 'last'},
                'search_space': [
                    {'param': 'Trainer.max_epochs', 'type': 'int', 'low': 1, 'high': 3},
                    {'param': 'Dataset.batch_size', 'type': 'categorical', 'choices': [16, 32]},
                ],
            }

    Info:
        * wrapped_software: Optuna, PyTorch Lightning (via ``TrainModel``)
    """

    def __init__(
        self,
        properties: dict,
        input_model_pth_path: str,
        input_dataset_pt_path: str,
        output_model_pth_path: str,
        **kwargs: Any,
    ) -> None:
        properties = properties or {}
        super().__init__(properties)

        self.input_model_pth_path = input_model_pth_path
        self.input_dataset_pt_path = input_dataset_pt_path
        self.output_model_pth_path = output_model_pth_path
        self.props = properties.copy()
        self.locals_var_dict = locals().copy()
        # Satisfy BiobbObject.check_properties (keys must exist on ``self``)
        self.Trainer = self.props.get("Trainer", {})
        self.Dataset = self.props.get("Dataset", {})
        self.Optuna = self.props.get("Optuna", {})

        self.io_dict = {
            "in": {
                "input_model_pth_path": input_model_pth_path,
                "input_dataset_pt_path": input_dataset_pt_path,
            },
            "out": {"output_model_pth_path": output_model_pth_path},
        }

        self.check_properties(properties)
        self.check_arguments()

    def _optuna_block(self) -> Dict[str, Any]:
        block = self.props.get("Optuna")
        if not isinstance(block, dict):
            raise KeyError("properties must contain an 'Optuna' dict")
        if "n_trials" not in block:
            raise KeyError("properties['Optuna'] must define 'n_trials'")
        if "search_space" not in block:
            raise KeyError("properties['Optuna'] must define 'search_space'")
        validate_search_space(block["search_space"])
        return block

    @launchlogger
    def launch(self) -> int:
        if self.check_restart():
            return 0

        self.stage_files()

        model_in = self.io_dict["in"]["input_model_pth_path"]
        data_in = self.io_dict["in"]["input_dataset_pt_path"]
        model_out = self.io_dict["out"]["output_model_pth_path"]

        optuna_cfg = self._optuna_block()
        n_trials = int(optuna_cfg["n_trials"])
        search_space = optuna_cfg["search_space"]

        unique_dir = self.stage_io_dict.get("unique_dir") or os.getcwd()
        trials_root = os.path.join(unique_dir, "optuna_trials")
        os.makedirs(trials_root, exist_ok=True)

        fu.log("## BioBB Hyperparameter tuning (Optuna) ##", self.out_log)
        fu.log(f"Input model: {os.path.abspath(model_in)}", self.out_log)
        fu.log(f"Input dataset: {os.path.abspath(data_in)}", self.out_log)
        fu.log(f"Trials: {n_trials}", self.out_log)

        study = create_study_from_config(optuna_cfg)
        objective = make_objective(
            base_properties=self.props,
            search_space=search_space,
            input_model_pth_path=model_in,
            input_dataset_pt_path=data_in,
            trials_root=trials_root,
            optuna_cfg=optuna_cfg,
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        ckpt = best.user_attrs.get("checkpoint")
        if not ckpt or not os.path.isfile(ckpt):
            shutil.rmtree(trials_root, ignore_errors=True)
            raise RuntimeError("Best trial has no saved checkpoint path")

        staged_out = self.stage_io_dict["out"]["output_model_pth_path"]
        shutil.copy2(ckpt, staged_out)
        fu.log(f"Best trial {best.number} value={best.value}", self.out_log)
        fu.log(f"Best model staged at {os.path.abspath(staged_out)}", self.out_log)
        fu.log(f"File size: {get_size(staged_out)}", self.out_log)

        shutil.rmtree(trials_root, ignore_errors=True)

        self.copy_to_host()
        self.remove_tmp_files()
        self.check_arguments(output_files_created=True, raise_exception=False)
        return 0


def hparam_tuning(
    properties: dict,
    input_model_pth_path: str,
    input_dataset_pt_path: str,
    output_model_pth_path: str,
    **kwargs: Any,
) -> int:
    """Run :class:`HparamTuning` and :meth:`launch`."""
    return HparamTuning(**dict(locals())).launch()


hparam_tuning.__doc__ = HparamTuning.__doc__
main = HparamTuning.get_main(
    hparam_tuning,
    "Hyperparameter search with Optuna for MDAE training.",
)
