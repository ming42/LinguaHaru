import logging
from .high_level import extract_and_translate, write_translated_result

log = logging.getLogger(__name__)

__version__ = "1.8.8"
__author__ = "Byaidu"
__all__ = ["extract_and_translate", "write_translated_result"]
