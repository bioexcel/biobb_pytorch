import argparse
import json
import torch
import os

def parse_ndx(ndx_path, group_name):
    atoms = []
    with open(ndx_path, 'r') as f:
        in_group = False
        for line in f:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_group = line[1:-1].strip()
                in_group = (current_group == group_name)
            elif in_group and line:
                atoms.extend(int(x) for x in line.split())
    return atoms

def generate_features_from_stats(stats_pt, features_path='features.dat'):
    stats = torch.load(stats_pt)
    feat_lines = []
    arg_list = []
    dist_count = 1
    ang_count = 1
    tor_count = 1
    if 'positions' in stats:
        pos_atoms = stats['positions'].flatten().tolist()
        pos_atoms = sorted(pos_atoms)
        for atom in pos_atoms:
            feat_lines.append(f'p{int(atom)}: POSITION ATOM={int(atom)}')
            arg_list.extend([f'p{int(atom)}.x', f'p{int(atom)}.y', f'p{int(atom)}.z'])
    if 'distances' in stats:
        for pair in stats['distances']:
            a, b = map(int, pair.tolist())
            label = f'd{dist_count}'
            feat_lines.append(f'{label}: DISTANCE ATOMS={a},{b}')
            arg_list.append(label)
            dist_count += 1
    if 'angles' in stats:
        for triple in stats['angles']:
            a, b, c = map(int, triple.tolist())
            label = f'a{ang_count}'
            feat_lines.append(f'{label}: ANGLE ATOMS={a},{b},{c}')
            arg_list.append(label)
            ang_count += 1
    if 'torsions' in stats or 'dihedrals' in stats:
        key = 'torsions' if 'torsions' in stats else 'dihedrals'
        for quad in stats[key]:
            a, b, c, d = map(int, quad.tolist())
            label = f't{tor_count}'
            feat_lines.append(f'{label}: TORSION ATOMS={a},{b},{c},{d}')
            arg_list.append(label)
            tor_count += 1
    with open(features_path, 'w') as f:
        f.write('\n'.join(feat_lines) + '\n')
    return ','.join(arg_list)

def main():
    parser = argparse.ArgumentParser(description='Generate PLUMED input file from inputs.')
    parser.add_argument('--model_pth', type=str, help='Path to the model.pth file.')
    parser.add_argument('--output_model_ptc_path', type=str, default='model.ptc', help='Path to save the converted model.ptc file.')
    parser.add_argument('--ndx', type=str, default=None, help='Optional path to the NDX file.')
    parser.add_argument('--features', type=str, default=None, help='Optional path to features.dat file. If not provided, it will be generated.')
    parser.add_argument('--ref_pdb', type=str, default=None, help='Optional reference PDB file (used for Cartesian coordinates).')
    parser.add_argument('--stats_pt', type=str, default=None, help='Optional path to stats.pt file (used to construct features.dat for non-Cartesian).')
    parser.add_argument('--properties', type=str, default='properties.json', help='Path to properties JSON file.')
    parser.add_argument('--output', type=str, default='plumed.dat', help='Output path for the generated PLUMED file.')
    args_parser = parser.parse_args()

    # Load properties dictionary
    with open(args_parser.properties, 'r') as f:
        properties = json.load(f)

    ndx_group = properties.get('ndx_group', 'chA_&_C-alpha')
    include_energy = properties.get('include_energy', True)
    biased_commands = properties.get('biased', [])
    prints_dict = properties.get('prints', {'ARG': '*', 'STRIDE': 1, 'FILE': 'COLVAR'})

    # Determine if features.dat needs to be generated
    generate_features = args_parser.features is None
    include_features = generate_features or (args_parser.features is not None)

    arg = None
    if generate_features:
        if args_parser.ref_pdb and args_parser.ndx and args_parser.stats_pt is None:
            # Cartesian mode: Generate features.dat from NDX group atoms
            atoms = parse_ndx(args_parser.ndx, ndx_group)
            atoms_sorted = sorted(atoms)
            feat_lines = []
            arg_list = []
            for atom in atoms_sorted:
                feat_lines.append(f'p{atom}: POSITION ATOM={atom}')
                arg_list += [f'p{atom}.x', f'p{atom}.y', f'p{atom}.z']
            with open('features.dat', 'w') as f:
                f.write('\n'.join(feat_lines) + '\n')
            arg = ','.join(arg_list)
        elif args_parser.stats_pt:
            # Non-Cartesian or mixed: Generate from stats.pt
            arg = generate_features_from_stats(args_parser.stats_pt)
        else:
            raise ValueError('Either ref_pdb + ndx (for Cartesian) or stats_pt (for others) required to generate features.dat.')
    else:
        # features.dat provided, require 'args' in properties
        if 'args' in properties:
            if isinstance(properties['args'], list):
                arg = ','.join(properties['args'])
            else:
                arg = properties['args']
        else:
            raise ValueError('If features.dat is provided, "args" must be specified in properties as a comma-separated string or list.')

    # Convert model.pth to model.ptc
    model_ptc = args_parser.output_model_ptc_path
    model = torch.load(args_parser.model_pth)
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, model_ptc)
    except Exception as e:
        print(f'jit.script failed: {e}. Attempting jit.trace instead.')
        num_inputs = len(arg.split(','))
        example_input = torch.randn(1, num_inputs)  # Assuming batch size 1, flat input
        traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(traced_model, model_ptc)

    # Build PLUMED lines
    plumed_lines = []
    if include_energy:
        plumed_lines.append('ene: ENERGY')
    if include_features:
        plumed_lines.append('INCLUDE FILE=features.dat')
    if args_parser.ndx:
        plumed_lines.append(f'c_alphas: GROUP NDX_FILE={args_parser.ndx} NDX_GROUP={ndx_group}')
    if args_parser.ref_pdb:
        if args_parser.ndx:
            plumed_lines.append('WHOLEMOLECULES ENTITY0=c_alphas')
            plumed_lines.append(f'FIT_TO_TEMPLATE STRIDE=1 REFERENCE={args_parser.ref_pdb} TYPE=OPTIMAL')
        else:
            print('Warning: Reference PDB provided but no NDX file; skipping WHOLEMOLECULES and FIT_TO_TEMPLATE.')
    plumed_lines.append(f'cv: PYTORCH_MODEL FILE={model_ptc} ARG={arg}')

    # Add biased dynamics commands + restraints
    for command in biased_commands:
        label = command.get('label', '')
        if label:
            label += ': '
        name = command['name']
        params_str = ' '.join(f'{k}={v}' for k, v in command.get('params', {}).items())
        plumed_lines.append(f'{label}{name} {params_str}')

    # Add prints
    prints_str = ' '.join(f'{k}={v}' for k, v in prints_dict.items())
    plumed_lines.append(f'PRINT {prints_str}')

    # Write to output file
    with open(args_parser.output, 'w') as f:
        f.write('\n'.join(plumed_lines) + '\n')

    print(f'PLUMED file generated at {args_parser.output}')

if __name__ == '__main__':
    main()