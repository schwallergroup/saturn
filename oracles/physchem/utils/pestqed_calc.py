"""
Functions and constants used to compute PestQED (for insecticides) based on RDKit's way of computing QED. 
Minimal modification of the actual QED module based on 10.1186/s13321-014-0042-6.
"""
import math

from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors as rdmd
from collections import namedtuple

QEDproperties = namedtuple('QEDproperties', 'MW,ALOGP,HBA,HBD,ROTB,AROM')
ADSparameter = namedtuple('ADSparameter', 'A,B,C,O, DMAX')

AliphaticRings = Chem.MolFromSmarts('[$([A;R][!a])]')


WEIGHT_NONE = QEDproperties(1.00, 1.00, 1.00, 1.00, 1.00, 1.00)

AcceptorSmarts = [
  '[oH0;X2]',
  '[OH1;X2;v2]',
  '[OH0;X2;v2]',
  '[OH0;X1;v2]',
  '[O-;X1]',
  '[SH0;X2;v2]',
  '[SH0;X1;v2]',
  '[S-;X1]',
  '[nH0;X2]',
  '[NH0;X1;v3]',
  '[$([N;+0;X3;v3]);!$(N[C,S]=O)]'
]
Acceptors = [Chem.MolFromSmarts(hba) for hba in AcceptorSmarts]

adsParameters = {
  'MW': ADSparameter(A=7.638E+001, B=2.983E+002, C=8.364E+001, O=1.912E+000, DMAX=78.2919965),
  'ALOGP': ADSparameter(A=7.427E+001, B=4.555E+000, C=-2.193E+000, O=-2.987E+000, DMAX=71.2829691),
  'HBA': ADSparameter(A=1.394E+002, B=1.363E+000, C=1.283E+000, O=5.341E-001,  DMAX=133.9224801),
  'HBD': ADSparameter(A=6.706E+002,  B=-1.163E+000,  C=7.856E-001, O=7.951E-001,  DMAX=331.170104),
  'ROTB': ADSparameter(A=6.549E+001,  B=6.219E+000,  C=-2.448E+000,  O=5.318E+000, DMAX=70.5540709),
  'AROM': ADSparameter(A=2.875E+002,  B=3.050E-001, C=1.554E+000,  O=-8.864E+001, DMAX=193.0023343)
}


def ads(x, adsParameter):
   p = adsParameter
   inner_exponent = -1.0 * math.exp(-1.0 * (x - p.B) / p.C)
   outer_exponent = math.exp(inner_exponent - (x - p.B) / p.C + 1.0)
   res = p.A * outer_exponent + p.O
   return res / p.DMAX


def properties(mol):
  """
  Calculates the properties that are required to calculate the PestQED descriptor.
  """
  if mol is None:
    raise ValueError('You need to provide a mol argument.')
  mol = Chem.RemoveHs(mol)

  qedProperties = QEDproperties(
    MW=rdmd._CalcMolWt(mol),
    ALOGP=Crippen.MolLogP(mol),
    HBA=sum(len(mol.GetSubstructMatches(pattern)) for pattern in Acceptors
            if mol.HasSubstructMatch(pattern)),
    HBD=rdmd.CalcNumHBD(mol),
    ROTB=rdmd.CalcNumRotatableBonds(mol, rdmd.NumRotatableBondsOptions.Strict),
    AROM=len(Chem.GetSSSR(Chem.DeleteSubstructs(Chem.Mol(mol), AliphaticRings))),
  )

  return qedProperties


def pestqed(mol, w=WEIGHT_NONE, qedProperties=None):
  """ 
  Calculate the weighted sum of ADS mapped properties

  some examples from the QED paper, reference values from Peter G's original implementation
  >>> m = Chem.MolFromSmiles('N=C(CCSCc1csc(N=C(N)N)n1)NS(N)(=O)=O')
  >>> qed(m)
  0.253...
  >>> m = Chem.MolFromSmiles('CNC(=NCCSCc1nc[nH]c1C)NC#N')
  >>> qed(m)
  0.234...
  >>> m = Chem.MolFromSmiles('CCCCCNC(=N)NN=Cc1c[nH]c2ccc(CO)cc12')
  >>> qed(m)
  0.234...
  """
  if qedProperties is None:
      qedProperties = properties(mol)

  d = [ads(pi, adsParameters[name]) for name, pi in qedProperties._asdict().items()]
  #clipping values so we don't get error
  d = [di if di > 0 else 0.00000001 for di in d]
  t = sum(wi * math.log(di) for wi, di in zip(w, d))
  return math.exp(t / sum(w))


if __name__ == "__main__":

    test = ["ClC1=CC=C(OP(CC)(SC(C)(CC)C)=O)C=C1",
            "FC(Cl)(SN(C(O/N=C(C)\SCC)=O)C)Cl"]

    mols = [Chem.MolFromSmiles(sm) for sm in test]

    qeis = [pestqed(mol) for mol in mols]

    print(qeis)
