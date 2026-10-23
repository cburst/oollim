"""Versioned, public-trained scoring choices shared by CLI and desktop."""
from .common import ROOT
FAMILIES = ('speechocean','korean','aesop','koreanocean','aesopocean','koreanaesop','aesopkoreanocean')
SELECTIONS = {
    'spearman': 'Spearman',
    'damae': 'DAMAE',
    'historical_grid_damae': 'Historical-grid DAMAE',
}
DEFAULT_SELECTION = 'spearman'

def bundle_path(family, selection=DEFAULT_SELECTION):
    if family not in FAMILIES or selection not in SELECTIONS:
        raise ValueError('Choose a supported family and selection rule.')
    return ROOT/'models'/'selection-comparison-v1'/selection/family/'family.json'
