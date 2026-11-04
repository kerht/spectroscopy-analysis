"""The Project class: a collection of measurements with loaders and project-level analysis."""

from .raman import ProjectRamanMixin
from .pl import ProjectPLMixin
from .trpl import ProjectTRPLMixin
from .io import ProjectIOMixin


class Project(ProjectIOMixin, ProjectRamanMixin, ProjectPLMixin, ProjectTRPLMixin):
    """Collection of measurements, usually one sample or one measurement folder.

    Attributes
    ----------
    data : dict
        Measurement name -> :class:`Data`.
    tables : dict
        Date -> lab register table (pandas DataFrame) read by ``open_table``.
    sample : str
        Sample name (set by the TRPL loaders).
    info : pandas Series or DataFrame
        Metadata of the last loaded measurement.
    """

    def __init__(self):
        self.tables = {}
        self.data = {}
        self.sample = ''
