import sys
import types

def apply_transformers_patch():
    try:
        import transformers.utils.import_utils as utils
        if not hasattr(utils, 'is_torch_fx_available'):
            utils.is_torch_fx_available = lambda: False
    except (ImportError, AttributeError):
        import transformers
        if not hasattr(transformers, 'utils'):
            transformers.utils = types.ModuleType('transformers.utils')
        
        utils = types.ModuleType('transformers.utils.import_utils')
        utils.is_torch_fx_available = lambda: False
        sys.modules['transformers.utils.import_utils'] = utils