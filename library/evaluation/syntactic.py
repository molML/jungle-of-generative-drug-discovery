from typing import List

import rdkit
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

rdkit.RDLogger.DisableLog("rdApp.*")


def clean_design(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    mol = uncharger.uncharge(mol)
    sanitization_flag = Chem.SanitizeMol(
        mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL, catchErrors=True
    )
    # SANITIZE_NONE is the "no error" flag of rdkit!
    if sanitization_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    can_smiles = Chem.MolToSmiles(mol, canonical=True)
    if can_smiles is None or len(can_smiles) == 0:
        return None

    return can_smiles


def get_valid_designs(design_list: List[str]) -> List[str]:
    cleaned_designs = [clean_design(design) for design in design_list]
    return [design for design in cleaned_designs if design is not None]
