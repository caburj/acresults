import numpy as np
import pandas as pd
import os

from .helper import *
from .constants import *

class ACResults:
    """
    The class abstracting all the results of an AquaCrop simulation.
    """
    def __init__(self, project_name, aquacrop_dir):
        self.fnames = get_result_fnames(project_name, aquacrop_dir)
        self.clim = None
        self.compec = None
        self.compwc = None
        self.crop = None
        self.inet = None
        self.prof = None
        self.run = None
        self.salt = None
        self.wabal = None

    def get(self, name, run_number=None):
        if name == "Clim":
            return self._get_Clim(run_number)

        if name == "CompEC":
            return self._get_CompEC(run_number)

        if name == "CompWC":
            return self._get_CompWC(run_number)

        if name == "Crop":
            return self._get_Crop(run_number)

        if name == "Inet":
            return self._get_Inet(run_number)

        if name == "Prof":
            return self._get_Prof(run_number)

        if name == "Salt":
            return self._get_Salt(run_number)

        if name == "Wabal":
            return self._get_Wabal(run_number)

        if name == "Run":
            return self._get_Run()
           

    def _get_Clim(self, run_number=None):
        if self.clim is None:
            self.clim = load(self.fnames['Clim'])
        
        if run_number is None:
            return self.clim

        return self.clim[run_number]

    def _get_CompEC(self, run_number=None):
        raise NotImplementedError()

        if self.compec is None:
            self.compec = load(self.fnames['CompEC'])

        if run_number is None:
            return self.compec

        return self.compec[run_number]

    def _get_CompWC(self, run_number=None):
        raise NotImplementedError()

        if self.compwc is None:
            self.compwc = load(self.fnames['CompWC'])

        if run_number is None:
            return self.compwc

        return self.compwc[run_number]

    def _get_Crop(self, run_number=None):
        if self.crop is None:
            self.crop = load(self.fnames['Crop'])

        if run_number is None:
            return self.crop

        return self.crop[run_number]
    
    def _get_Inet(self, run_number=None):
        if self.inet is None:
            self.inet = load(self.fnames['Inet'])

        if run_number is None:
            return self.inet

        return self.inet[run_number]

    def _get_Prof(self, run_number=None):
        if self.prof is None:
            self.prof = load(self.fnames['Prof'])

        if run_number is None:
            return self.prof

        return self.prof[run_number]

    def _get_Salt(self, run_number=None):
        if self.salt is None:
            self.salt = load(self.fnames['Salt'])

        if run_number is None:
            return self.salt

        return self.salt[run_number]

    def _get_Wabal(self, run_number=None):
        if self.wabal is None:
            self.wabal = load(self.fnames['Wabal'])

        if run_number is None:
            return self.wabal

        return self.wabal[run_number]

    def _get_Run(self):
        if self.run is None:
            self.run = load_Run(self.fnames['Run'])
        return self.run
