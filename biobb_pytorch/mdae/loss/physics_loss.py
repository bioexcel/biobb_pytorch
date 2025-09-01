import torch
import numpy as np
from biobb_pytorch.mdae.loss.utils.torch_protein_energy import TorchProteinEnergy

class PhysicsLoss(torch.nn.Module):
    """
    Physics loss for the FoldingNet model.
    """

    def __init__(self, stats, protein_energy=None, physics_scaling_factor=0.1):
        super().__init__()

        if stats is not None:
            top = stats['topology']

            x0_coords = torch.tensor(top.xyz[0]).permute(1, 0)

            atominfo = []
            for i in top.topology.atoms:
                atominfo.append([i.name, i.residue.name, i.residue.index+1])
            atominfo = np.array(atominfo, dtype=object)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.protein_energy = TorchProteinEnergy(x0_coords,
                                                    atominfo, 
                                                    device=device, 
                                                    method='roll')
            
        else:
            self.protein_energy = protein_energy

        self.psf = physics_scaling_factor

    def mse_loss(self, batch, decoded):
        """
        Mean squared error loss for the FoldingNet model.
        """
        return ((batch - decoded) ** 2).mean()
    
    def total_physics_loss(self, decoded_interpolation):
        '''
        Called from both :func:`train_step <molearn.trainers.Torch_Physics_Trainer.train_step>` and :func:`valid_step <molearn.trainers.Torch_Physics_Trainer.valid_step>`.
        Takes random interpolations between adjacent samples latent vectors. These are decoded (decoded structures saved as ``self._internal['generated'] = generated if needed elsewhere) and the energy terms calculated with ``self.physics_loss``.

        :param torch.Tensor batch: tensor of shape [batch_size, 3, n_atoms]. Give access to the mini-batch of structures. This is used to determine ``n_atoms``
        :param torch.Tensor latent: tensor shape [batch_size, 2, 1]. Pass the encoded vectors of the mini-batch.
        '''
        bond, angle, torsion = self.protein_energy._roll_bond_angle_torsion_loss(decoded_interpolation)
        n = len(decoded_interpolation)
        bond/=n
        angle/=n
        torsion/=n
        _all = torch.tensor([bond, angle, torsion])
        _all[_all.isinf()]=1e35
        total_physics = _all.nansum()

        return {'physics_loss':total_physics, 'bond_energy':bond, 'angle_energy':angle, 'torsion_energy':torsion}
    
    def forward(self, 
                batch, 
                decoded, 
                decoded_interpolation
        ):
        """
        Forward pass for the FoldingNet model.
        """
        mse_loss = self.mse_loss(batch, decoded)

        physics_loss_dict = self.total_physics_loss(decoded_interpolation)
        physics_loss = physics_loss_dict['physics_loss']

        with torch.no_grad():
            scale = self.psf*mse_loss/(physics_loss+1e-5)

        return mse_loss, physics_loss_dict, scale