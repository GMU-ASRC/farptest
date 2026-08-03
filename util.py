import os
from pathlib import Path
from warnings import warn
from swarmsim.util.processing.multicoreprocessing import process_map


def glob_import(domain, glob, path=None, error='warn'):
    modules = {}
    for f in Path('.' if path is None else path).glob(glob):
        try:
            module, controller = load_cls_from_file(f, domain)
            modules[f] = (module, controller)
        except ImportError as err:
            if error == 'warn':
                warn(err.msg, stacklevel=1)
            elif error == 'raise':
                raise err
    return modules


def load_cls_from_file(path: os.PathLike | str, namespace: str):
    # only works if class has same name as file
    from importlib.util import spec_from_file_location, module_from_spec
    from swarmsim import register_dictlike_type

    f = Path(path)
    assert f.exists()

    spec = spec_from_file_location(f.stem, f)
    assert spec is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, module.__name__)
    register_dictlike_type(namespace, cls.__name__, cls)
    return module, cls


def load_all_controllers(path=None, error='warn', name_suffix='Controller'):
    return glob_import('controller', f'*{name_suffix}.py', path, error)


def load_all_sensors(path=None, error='warn', name_suffix='Sensor'):
    return glob_import('sensor', f'*{name_suffix}.py', path, error)


def test_mp(configs, func, tqdm_kwargs={}):
    ret_arr = process_map(func, configs, **tqdm_kwargs)
    stats, successes = zip(*ret_arr)

    rate = 1 - sum(successes) / len(configs)
    return stats, rate
