
from genericpath import getsize
import torch
from typing import Dict, Any, Optional, List
import os
import argparse
from biobb_pytorch.mdae.utils.log_utils import get_size
from biobb_common.configuration import settings
from biobb_common.tools.file_utils import launchlogger
from biobb_common.tools import file_utils as fu
from biobb_common.generic.biobb_object import BiobbObject


class GeneratePlumed(BiobbObject):
    """
    | biobb_plumed GeneratePlumed
    | Generates a PLUMED input file for biased dynamics using a PyTorch model.
    | Generates a PLUMED input file, features.dat, and converts the model to .ptc format.

    Args:
        input_model_pth_path (str): Path to the input PyTorch model file (.pth). File type: input. Accepted formats: pth (edam:format_2333).
        input_stats_pt_path (str) (Optional): Path to the input statistics file (.pt) for non-Cartesian features. File type: input. Accepted formats: pt (edam:format_2333).
        input_reference_pdb_path (str) (Optional): Path to the reference PDB file for Cartesian mode. File type: input. Accepted formats: pdb (edam:format_1476).
        input_ndx_path (str) (Optional): Path to the GROMACS NDX file. File type: input. Accepted formats: ndx (edam:format_2033).
        output_plumed_dat_path (str): Path to the output PLUMED file. File type: output. Accepted formats: dat (edam:format_2330).
        output_features_dat_path (str): Path to the output features.dat file. File type: output. Accepted formats: dat (edam:format_2330).
        output_model_ptc_path (str): Path to the output TorchScript model file (.ptc). File type: output. Accepted formats: ptc (edam:format_2333).
        properties (dict - Python dictionary object containing the tool parameters, not input/output files):
            * **include_energy** (*bool*) - (True) Whether to include ENERGY in PLUMED.
            * **biased** (*list*) - ([]) List of biased dynamics commands.
            * **prints** (*dict*) - ({"ARG": "*", "STRIDE": 1, "FILE": "COLVAR"}) PRINT command parameters.
            * **group** (*dict*) - (None) GROUP command parameters.
            * **wholemolecules** (*dict*) - (None) WHOLEMOLECULES command parameters.
            * **fit_to_template** (*dict*) - (None) FIT_TO_TEMPLATE command parameters.
            * **pytorch_model** (*dict*) - (None) PYTORCH_MODEL command parameters.

    Examples:
        This is a use case of how to use the building block from Python:

            from biobb_plumed.generate_plumed import generatePlumed

            prop = {
                "additional_actions": [
                    {
                        "name": "ENERGY",
                        "label": "ene"
                    },
                    {
                        "name": "RMSD",
                        "label": "rmsd",
                        "params": {
                            "TYPE": "OPTIMAL"
                        }
                    }
                ],
                "group": {
                    "label": "c_alphas",
                    "NDX_GROUP": "chA_&_C-alpha"
                },
                "wholemolecules": {
                    "ENTITY0": "c_alphas"
                },
                "fit_to_template": {
                    "STRIDE": 1,
                    "TYPE": "OPTIMAL"
                },
                "pytorch_model": {
                    "label": "cv",
                    "PACE": 1
                },
                "biased": [
                    {
                        "name": "METAD",
                        "label": "bias",
                        "params": {
                            "ARG": "cv.1",
                            "PACE": 500,
                            "HEIGHT": 1.2,
                            "SIGMA": 0.35,
                            "FILE": "HILLS",
                            "BIASFACTOR": 8
                        }
                    }
                ],
                "prints": {
                    "ARG": "cv.*,bias.*",
                    "STRIDE": 1,
                    "FILE": "COLVAR"
                }
            }

            generatePlumed(
                input_model_pth_path="model.pth",
                input_stats_pt_path="stats.pt",
                output_plumed_dat_path="plumed.dat",
                output_features_dat_path="features.dat",
                output_model_ptc_path="model.ptc",
                properties=prop
            )

    Info:
        * wrapped_software:
            * name: PLUMED with PyTorch
            * version: >=2.0
            * license: LGPL
        * ontology:
            * name: EDAM
            * schema: http://edamontology.org/EDAM.owl
    """

    def __init__(
        self,
        input_model_pth_path: str,
        input_stats_pt_path: Optional[str] = None,
        input_reference_pdb_path: Optional[str] = None,
        input_ndx_path: Optional[str] = None,
        output_plumed_dat_path: str = 'plumed.dat',
        output_features_dat_path: str = 'features.dat',
        output_model_ptc_path: str = 'model.ptc',
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        properties = properties or {}

        super().__init__(properties)
        self.locals_var_dict = locals().copy()

        # Input/Output files
        self.io_dict = {
            "in": {"input_model_pth_path": input_model_pth_path},
            "out": {
                "output_plumed_dat_path": output_plumed_dat_path,
                "output_features_dat_path": output_features_dat_path,
                "output_model_ptc_path": output_model_ptc_path
            }
        }
        if input_stats_pt_path:
            self.io_dict["in"]["input_stats_pt_path"] = input_stats_pt_path
        if input_reference_pdb_path:
            self.io_dict["in"]["input_reference_pdb_path"] = input_reference_pdb_path
        if input_ndx_path:
            self.io_dict["in"]["input_ndx_path"] = input_ndx_path

        # Properties
        self.model_pth = input_model_pth_path
        self.stats_pt = input_stats_pt_path
        self.ref_pdb = input_reference_pdb_path
        self.ndx = input_ndx_path
        self.properties = properties

        self.additional_actions = self.properties.get('additional_actions', [])
        self.group = self.properties.get('group', None)
        self.wholemolecules = self.properties.get('wholemolecules', None)
        self.fit_to_template = self.properties.get('fit_to_template', None)
        self.pytorch_model = self.properties.get('pytorch_model', None)
        self.bias = self.properties.get('bias', [])
        self.prints = self.properties.get('prints', {'ARG': '*', 'STRIDE': 1, 'FILE': 'COLVAR'})

        # Check the properties
        self.check_properties(properties)
        self.check_arguments()

        self.stats = self._load_stats()
        self.n_features = self.stats.get('shape', [None, None])[1]
        self.arg = ','.join(self._generate_features()[1])
        self._convert_model_to_ptc()

    def _load_stats(self) -> Optional[Dict[str, Any]]:
        """Load stats.pt if provided."""
        if self.stats_pt:
            return torch.load(self.stats_pt)
        return None

    def _generate_features(self) -> str:
        """
        Generate features.dat and return the ARG string for PYTORCH_MODEL.

        Returns:
            str: Comma-separated ARG string.
        """
        if self.stats_pt:
            # Non-Cartesian or mixed mode
            return self._generate_features_from_stats(self.stats, self.io_dict['out']['output_features_dat_path'])
        else:
            raise ValueError('Input_stats_pt_path is required.')

    def _generate_features_from_stats(self, stats: Dict[str, Any], features_path: str) -> str:
        """
        Generate features.dat from stats.pt for distances, angles, dihedrals, and/or cartesians.

        Args:
            stats (Dict[str, Any]): Loaded stats dictionary.
            features_path (str): Path to write features.dat.

        Returns:
            str: Comma-separated ARG string.
        """
        feat_lines = []
        arg_list = []
        dist_count = 1
        ang_count = 1
        tor_count = 1

        # Adjust indices to 1-based for PLUMED
        def adjust_indices(indices: List[int]) -> List[int]:
            return [idx + 1 for idx in indices]

        if 'cartesian_indices' in stats:
            pos_atoms = adjust_indices(stats['cartesian_indices'])
            fu.log(f"Found {len(pos_atoms)} Cartesian features.", self.out_log)
            for atom in pos_atoms:
                feat_lines.append(f'p{atom}: POSITION ATOM={atom}')
                arg_list.extend([f'p{atom}.x', f'p{atom}.y', f'p{atom}.z'])

        if 'distance_indices' in stats:
            fu.log(f"Found {len(stats['distance_indices'])} Distance features.", self.out_log)
            for pair in stats['distance_indices']:
                a, b = adjust_indices(pair)
                label = f'd{dist_count}'
                feat_lines.append(f'{label}: DISTANCE ATOMS={a},{b}')
                arg_list.append(label)
                dist_count += 1

        if 'angle_indices' in stats:
            fu.log(f"Found {len(stats['angle_indices'])} Angle features.", self.out_log)
            for triple in stats['angle_indices']:
                a, b, c = adjust_indices(triple)
                label = f'a{ang_count}'
                feat_lines.append(f'{label}: ANGLE ATOMS={a},{b},{c}')
                arg_list.append(label)
                ang_count += 1

        if 'dihedral_indices' in stats:
            fu.log(f"Found {len(stats['dihedral_indices'])} Dihedral features.", self.out_log)
            for quad in stats['dihedral_indices']:
                a, b, c, d = adjust_indices(quad)
                label = f't{tor_count}'
                feat_lines.append(f'{label}: TORSION ATOMS={a},{b},{c},{d}')
                arg_list.append(label)
                tor_count += 1

        return feat_lines, arg_list

    # def _parse_ndx(self, ndx_path: str, group_name: str) -> List[int]:
    #     """Parse atom indices from a GROMACS NDX file for a specific group."""
    #     atoms = []
    #     with open(ndx_path, 'r') as f:
    #         in_group = False
    #         for line in f:
    #             line = line.strip()
    #             if line.startswith('[') and line.endswith(']'):
    #                 current_group = line[1:-1].strip()
    #                 in_group = (current_group == group_name)
    #             elif in_group and line:
    #                 atoms.extend(int(x) for x in line.split())
    #     fu.log(f'Parsed {len(atoms)} atoms from NDX group "{group_name}" in {ndx_path}', self.out_log)
    #     return atoms

    # def _get_chain_from_group(self, group_name: str) -> str:
    #     """Extract chain identifier from NDX group name."""
    #     if group_name.startswith('ch') and '_' in group_name:
    #         chain_part = group_name[2:group_name.index('_')]
    #         if len(chain_part) == 1:
    #             return chain_part
    #     return 'A'  # Default chain

    # def _generate_ndx_from_pdb(self, pdb_path: str, group_name: str) -> str:
    #     """
    #     Generate NDX file from PDB by extracting C-alpha atoms in the specified chain.

    #     Args:
    #         pdb_path (str): Path to PDB file.
    #         group_name (str): NDX group name.

    #     Returns:
    #         str: Path to generated NDX file.
    #     """
    #     chain = self._get_chain_from_group(group_name)
    #     atoms = []
    #     with open(pdb_path, 'r') as f:
    #         for line in f:
    #             if line.startswith('ATOM'):
    #                 atom_name = line[12:16].strip()
    #                 chain_id = line[21]
    #                 if atom_name == 'CA' and chain_id == chain:
    #                     atom_num = int(line[6:11].strip())
    #                     atoms.append(atom_num)
    #     if not atoms:
    #         raise ValueError(f'No C-alpha atoms found in chain {chain} of {pdb_path}. Cannot generate NDX.')
    #     ndx_path = 'generated.ndx'
    #     with open(ndx_path, 'w') as f:
    #         f.write(f'[ {group_name} ]\n')
    #         for i, atom in enumerate(atoms, 1):
    #             f.write(f'{atom} ')
    #             if i % 15 == 0:
    #                 f.write('\n')
    #         if len(atoms) % 15 != 0:
    #             f.write('\n')
    #     fu.log(f'Generated NDX file with {len(atoms)} atoms at {ndx_path}', self.out_log)
    #     return ndx_path

    def _convert_model_to_ptc(self) -> None:
        """Convert the PyTorch model to TorchScript format (.ptc)."""
        model = torch.load(self.model_pth)
        self._enable_jit_scripting(model)
        output_path = self.io_dict['out']['output_model_ptc_path']
        try:
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, output_path)
            fu.log(f'Successfully scripted and saved model to {output_path}', self.out_log)
        except Exception as e:
            fu.log(f'jit.script failed: {e}. Attempting jit.trace instead.', self.out_log)
            example_input = torch.randn(1, self.n_features)  # Batch size 1, flat input
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, output_path)
            fu.log(f'Successfully traced and saved model to {output_path}', self.out_log)

    def _enable_jit_scripting(self, module: torch.nn.Module) -> None:
        """Set _jit_is_scripting flag to True for the module and submodules to enable scripting."""
        if hasattr(module, '_jit_is_scripting'):
            module._jit_is_scripting = True
        for subm in module.modules():
            if hasattr(subm, '_jit_is_scripting'):
                subm._jit_is_scripting = True

    def _build_plumed_lines(self) -> List[str]:
        """Build the list of lines for the PLUMED file."""
        lines = []
        lines.append(f'INCLUDE FILE={os.path.abspath(self.io_dict["out"]["output_features_dat_path"])}')

        # Additional actions (e.g., ENERGY, other metrics)
        for action in self.additional_actions:
            label = action.get('label', '')
            if label:
                label += ': '
            name = action['name']
            params_str = ' '.join(f'{k}={v}' for k, v in action.get('params', {}).items())
            lines.append(f'{label}{name} {params_str}')

        # GROUP
        group_label = 'C-alpha'
        if self.group:
            g = self.group
            group_label = g.get('label', 'C-alpha')
            params = ' '.join(f'{k}={v}' for k, v in g.items() if k not in ['label', 'name'])
            lines.append(f"{group_label}: GROUP {params}")
            fu.log(f'Using GROUP: {group_label}', self.out_log)
            fu.log(f'   Parameters:', self.out_log)
            for k, v in g.items():
                if k not in ['label', 'name']:
                    fu.log(f'    > {k.upper()}: {v}', self.out_log)

        # WHOLEMOLECULES
        uses_positions = True if 'cartesian_indices' in self.stats else False
        if uses_positions:
            if self.wholemolecules: 
                w = self.wholemolecules
                params = ' '.join(f'{k}={v}' for k, v in w.items())
                lines.append(f'WHOLEMOLECULES {params}')
                fu.log(f'Using WHOLEMOLECULES with parameters: {params}', self.out_log)
            else:
                fu.log('WARNING: Using Cartesian coordinates but no WHOLEMOLECULES parameters provided; add WHOLEMOLECULES in properties.', self.out_log)
        else:
            if self.wholemolecules:
                fu.log('NOTE: Reference PDB provided but no POSITION features detected; skipping WHOLEMOLECULES.', self.out_log)

        # FIT_TO_TEMPLATE
        if uses_positions:
            if self.fit_to_template:
                f = self.fit_to_template
                params = ' '.join(f'{k}={v}' for k, v in f.items())
                lines.append(f'FIT_TO_TEMPLATE REFERENCE={os.path.abspath(self.ref_pdb)} {params}')
                fu.log(f'Using FIT_TO_TEMPLATE', self.out_log)
                fu.log(f'   Reference PDB: {os.path.abspath(self.ref_pdb)}', self.out_log)
                fu.log(f'   Parameters:', self.out_log)
                for k, v in f.items():
                    fu.log(f'    > {k.upper()}: {v}', self.out_log)
            else:
                fu.log('WARNING: Using Cartesian coordinates but no FIT_TO_TEMPLATE parameters provided; add FIT_TO_TEMPLATE in properties.', self.out_log)
        else:
            if self.fit_to_template:
                fu.log('NOTE: Reference PDB provided but no POSITION features detected; skipping FIT_TO_TEMPLATE.', self.out_log)

        # PYTORCH_MODEL
        pyt_label = 'cv'
        pyt_params = {'FILE': os.path.abspath(self.io_dict['out']['output_model_ptc_path']), 'ARG': self.arg}
        if self.pytorch_model:
            p = self.pytorch_model
            pyt_label = p.get('label', 'cv')
            pyt_params.update({k: v for k, v in p.items() if k not in ['label']})
        params_str = ' '.join(f'{k}={v}' for k, v in pyt_params.items())
        params_non_args = {f'{k}: {v}' for k, v in pyt_params.items() if k != 'ARG'}
        lines.append(f'{pyt_label}: PYTORCH_MODEL {params_str}')
        fu.log(f'Using PYTORCH_MODEL: {pyt_label}', self.out_log)
        fu.log(f'   Model ptc file: {os.path.abspath(self.io_dict["out"]["output_model_ptc_path"])}', self.out_log)
        for param in params_non_args:
            if not param.startswith('FILE'):
                fu.log(f'   Parameters:', self.out_log)
                fu.log(f'    > {param}', self.out_log)

        # Bias actions
        for command in self.bias:
            label = command.get('label', '')
            if label:
                label += ': '
            name = command['name']
            params_str = ' '.join(f'{k}={v}' for k, v in command.get('params', {}).items())
            lines.append(f'{label}{name} {params_str}')
            fu.log(f'Using Bias:', self.out_log)
            fu.log(f'   Command: {name}', self.out_log)
            fu.log(f'   Parameters:', self.out_log)
            for param in command.get('params', {}).items():
                fu.log(f'    > {param[0]}: {param[1]}', self.out_log)
        # PRINT
        prints_str = ' '.join(f'{k}={v}' for k, v in self.prints.items())
        lines.append(f'PRINT {prints_str}')

        return lines

    @launchlogger
    def launch(self) -> int:
        """Execute the generation of PLUMED files."""

        # Setup Biobb
        if self.check_restart():
            return 0

        self.stage_files()

        features_lines = self._generate_features()[0]
        plumed_lines = self._build_plumed_lines()

        has_cartesian = True if 'cartesian_indices' in self.stats else False
        if self.ndx is None:
            if has_cartesian:
                fu.log('WARNING: When employing Cartesian coordinates as collective variables (CVs) for biasing in PLUMED, ' \
                    'an NDX index file is required to properly define atom groups for fitting and alignment purposes, ' \
                    'make sure to provide a NDX file.', self.out_log)
        
        with open(self.io_dict['out']['output_features_dat_path'], 'w') as f:
            for line in features_lines:
                f.write(line + '\n')
        fu.log(f'Generated features.dat at {os.path.abspath(self.io_dict["out"]["output_features_dat_path"])}', self.out_log)
        fu.log(f'File size: {get_size(self.io_dict["out"]["output_features_dat_path"])}', self.out_log)

        with open(self.io_dict['out']['output_plumed_dat_path'], 'w') as f:
            f.write('\n'.join(plumed_lines) + '\n')
        fu.log(f'Generated PLUMED file at {os.path.abspath(self.io_dict["out"]["output_plumed_dat_path"])}', self.out_log)
        fu.log(f'File size: {get_size(self.io_dict["out"]["output_plumed_dat_path"])}', self.out_log)

        # Copy files to host
        self.copy_to_host()

        # Remove temporal files
        self.remove_tmp_files()

        self.check_arguments(output_files_created=True, raise_exception=False)

        return 0

def generatePlumed(
    input_model_pth_path: str,
    input_stats_pt_path: Optional[str] = None,
    input_reference_pdb_path: Optional[str] = None,
    input_ndx_path: Optional[str] = None,
    output_plumed_dat_path: str = 'plumed.dat',
    output_features_dat_path: str = 'features.dat',
    output_model_ptc_path: str = 'model.ptc',
    properties: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Execute the :class:`GeneratePlumed <generate_plumed.GeneratePlumed>` class and
    execute the :meth:`launch() <generate_plumed.GeneratePlumed.launch>` method.
    """
    return GeneratePlumed(
        input_model_pth_path=input_model_pth_path,
        input_stats_pt_path=input_stats_pt_path,
        input_reference_pdb_path=input_reference_pdb_path,
        input_ndx_path=input_ndx_path,
        output_plumed_dat_path=output_plumed_dat_path,
        output_features_dat_path=output_features_dat_path,
        output_model_ptc_path=output_model_ptc_path,
        properties=properties,
    ).launch()

generatePlumed.__doc__ = GeneratePlumed.__doc__

def main():
    """Command line execution of this building block. Please check the command line documentation."""

    parser = argparse.ArgumentParser(
        description="Generates PLUMED files for biased dynamics.",
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, width=99999),
    )

    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="This file can be a YAML file, JSON file or JSON string",
    )

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "--input_model_pth_path",
        required=True,
        help="Input PyTorch model file path (.pth)"
    )
    parser.add_argument(
        "--input_stats_pt_path",
        required=False,
        help="Input statistics file path (.pt)"
    )
    parser.add_argument(
        "--input_reference_pdb_path",
        required=False,
        help="Reference PDB file path"
    )
    parser.add_argument(
        "--input_ndx_path",
        required=False,
        help="GROMACS NDX file path"
    )
    required_args.add_argument(
        "--output_plumed_dat_path",
        required=True,
        help="Output PLUMED file path"
    )
    required_args.add_argument(
        "--output_features_dat_path",
        required=True,
        help="Output features.dat file path"
    )
    required_args.add_argument(
        "--output_model_ptc_path",
        required=True,
        help="Output TorchScript model file path (.ptc)"
    )

    args = parser.parse_args()
    config = args.config if args.config else None
    properties = settings.ConfReader(config=config).get_prop_dic()

    generatePlumed(
        input_model_pth_path=args.input_model_pth_path,
        input_stats_pt_path=args.input_stats_pt_path,
        input_reference_pdb_path=args.input_reference_pdb_path,
        input_ndx_path=args.input_ndx_path,
        output_plumed_dat_path=args.output_plumed_dat_path,
        output_features_dat_path=args.output_features_dat_path,
        output_model_ptc_path=args.output_model_ptc_path,
        properties=properties,
    )

if __name__ == "__main__":
    main()