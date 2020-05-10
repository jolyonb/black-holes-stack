"""
common.py

Contains common components for the project. Includes:

* Persistence class
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import datetime
from typing import Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stack import Model

class Persistence(ABC):
    """Abstract base class to handle all persistence in the project"""

    def __init__(self, model: 'Model') -> None:
        """
        Initializes the persistence aspects of the class.
        
        :param model: Model class this object is computing data for
        """
        self.model = model
        self.timestamp = None  # Timestamp at which data last finished computing

    @property
    @abstractmethod
    def filename(self) -> str:
        """Returns the generic filename associated with this class"""
        raise NotImplementedError()

    @abstractmethod
    def load_data(self) -> None:
        """Load the data associated with this class"""
        raise NotImplementedError()

    @abstractmethod
    def compute_data(self) -> None:
        """General method to compute all data associated with this class"""
        raise NotImplementedError()

    @abstractmethod
    def save_data(self) -> None:
        """Save the data associated with this class"""
        raise NotImplementedError()

    @property
    def ready(self) -> bool:
        """Flag to indicate whether data for this class has been constructed"""
        if self.timestamp is None:
            return False
        return True

    def check_files(self, prev_timestamp: Optional[datetime.datetime]) -> bool:
        """
        Checks to see if the params file exists for the class, and is valid. Returns True if so, and False if not.
        
        :param prev_timestamp: Timestamp of previous step in computation. If the timestamp we read is before this,
                               return False (invalid).
        """
        if not self.file_exists(self.filename + '-params.txt'):
            return False
        
        timestamp = self.check_model_params()
        
        if timestamp is None or (prev_timestamp is not None and timestamp < prev_timestamp):
            return False
        
        return True
    
    def file_path(self, filename: str) -> str:
        """Construct the path for a file in this model"""
        return os.path.join(self.model.path, filename)

    def file_exists(self, filename: str) -> bool:
        """Checks to ensure that the given filename exists for the current model"""
        return os.path.isfile(self.file_path(filename))
    
    def construct_data(self,
                       prev_timestamp: Optional[datetime.datetime] = None,
                       recalculate: bool = False
                       ) -> None:
        """
        Either loads or constructs the data associated with this class.
        
        :param prev_timestamp: The timestamp for the computation of the previous step of the data.
        :param recalculate: Specifies whether data should be computed from scratch, ignoring any saved data.
        """
        if self.filename is None:
            # Used when no data needs to be persisted
            return

        # Determine whether to load or recompute data
        if recalculate or self.model.recompute_all or not self.check_files(prev_timestamp):
            recalculate = True
        
        if recalculate:
            if self.model.verbose:
                print('    Computing...')
            self.compute_data()
            self.save_data()
            self.save_model_params()
        else:
            if self.model.verbose:
                print('    Loading...')
            self.load_data()
            self.timestamp = self.check_model_params()  # Save timestamp only after successfully loading data

    def save_model_params(self) -> None:
        """
        Save the model parameters and timestamp for this computation.
        Uses the class filename to save the results.
        """
        self.timestamp = datetime.datetime.now()
        path = self.file_path(self.filename + '-params.txt')
        with open(path, 'w') as f:
            f.write(f'Timestamp: {self.timestamp}\n')
            f.write(f'n_efolds: {self.model.n_efolds}\n')
            f.write(f'n_fields: {self.model.n_fields}\n')
            f.write(f'mpsi: {self.model.mpsi}\n')
            f.write(f'm0: {self.model.m0}\n')
            f.write(f'min_k: {self.model.min_k}\n')
            f.write(f'num_modes: {self.model.num_modes}\n')
            f.write(f'max_k: {self.model.max_k}\n')
            f.write(f'test_ps: {self.model.test_ps}\n')

    def check_model_params(self) -> Union[datetime.datetime, None]:
        """
        Checks to see if saved model parameters for the previous computation agree with current
        model parameters.
        
        :return: Timestamp of previous computation, or None if not found or invalid
        """
        # Load data
        path = self.file_path(self.filename + '-params.txt')
        with open(path) as f:
            data = f.readlines()
        trimmed = [x.split(":", 1)[1].strip() for x in data]
        
        # Unpack and convert data into appropriate types
        timestamp, n_efolds, n_fields, mpsi, m0, min_k, num_modes, max_k, test_ps = trimmed
        n_efolds = float(n_efolds)
        n_fields = int(n_fields)
        mpsi = float(mpsi)
        m0 = float(m0)
        min_k = float(min_k)
        num_modes = int(num_modes)
        max_k = float(max_k)
        test_ps = bool(test_ps)
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')

        # Check data for agreement with model
        if n_efolds != self.model.n_efolds:
            return None
        if n_fields != self.model.n_fields:
            return None
        if mpsi != self.model.mpsi:
            return None
        if m0 != self.model.m0:
            return None
        if min_k != self.model.min_k:
            return None
        if num_modes != self.model.num_modes:
            return None
        if max_k != self.model.max_k:
            return None
        if test_ps != self.model.test_ps:
            return None

        return timestamp
