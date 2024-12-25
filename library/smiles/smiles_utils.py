import random
import re
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# _ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
_ELEMENTS_STR = r"(?<=\[)Cs(?=\])|\<BEG\>|\<PAD\>|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
__REGEXES = {
    "segmentation_sq": rf"(\[|\]|{_ELEMENTS_STR}|"
    + r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)",
}
_RE_PATTERNS = {name: re.compile(pattern) for name, pattern in __REGEXES.items()}


def segment_smiles(smiles: str, segment_sq_brackets=True) -> List[str]:
    regex = _RE_PATTERNS["segmentation_sq"]
    if not segment_sq_brackets:
        regex = _RE_PATTERNS["segmentation"]
    return regex.findall(smiles)


def segment_smiles_batch(
    smiles_batch: List[str], segment_sq_brackets=True
) -> List[List[str]]:
    return [segment_smiles(smiles, segment_sq_brackets) for smiles in smiles_batch]


def sanitize_smiles(
    smiles: str,
    to_canonical=True,
):
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

    return Chem.MolToSmiles(mol, canonical=to_canonical)


def sanitize_smiles_batch(
    smiles_batch: List[str],
    to_canonical=True,
):
    return [
        sanitize_smiles(
            smiles,
            to_canonical=to_canonical,
        )
        for smiles in smiles_batch
    ]


def enumerate(smiles: str, n_enumerations: int = 10, seed: int = None):
    if seed is not None:
        random.seed(seed)
    mol = Chem.MolFromSmiles(smiles)
    atom_indices = list(range(mol.GetNumAtoms()))
    enumerated_smiles = list()
    for _ in range(n_enumerations):
        random.shuffle(atom_indices)
        shuffled_mol = Chem.RenumberAtoms(mol, atom_indices)
        enumerated_smiles.append(
            Chem.MolToSmiles(shuffled_mol, canonical=False, isomericSmiles=True)
        )

    return list(set(enumerated_smiles))
