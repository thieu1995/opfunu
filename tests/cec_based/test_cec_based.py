import numpy as np

from opfunu import get_all_cec_functions


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_cec_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f'{f.__name__} failed to have ndim == dim_default'

def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct , values.
    known_failing = ['F72008', 'F222013', 'F232013', 'F242013', 'F252013',
                     'F132014', 'F142014', 'F182014', 'F192014', 'F202014', 'F212014', 'F222014',
                     'F242014', 'F252014', 'F272014', 'F282014', 'F292014', 'F302014', 'F112015',
                     'F122015', 'F132015', 'F142015', 'F152015', 'F62015', 'F72015', 'F102017', 'F122017',
                     'F142017', 'F152017', 'F162017', 'F172017', 'F182017', 'F192017', 'F202017', 'F212017',
                     'F222017', 'F232017', 'F242017', 'F252017', 'F262017', 'F272017', 'F282017', 'F292017',
                     'F82017', 'F12019', 'F22019', 'F32019', 'F92019', 'F102020', 'F62020', 'F72020', 'F82020',
                     'F92020', 'F102021', 'F62021', 'F72021', 'F82021', 'F92021', 'F102022', 'F112022',
                     'F122022', 'F52022', 'F62022', 'F72022', 'F82022']
    allFunctions = [x for x in get_all_cec_functions() if x.__name__ not in known_failing]
    for f in allFunctions:
        f_default = f()
        x_global = f_default.x_global
        assert abs(f_default.evaluate(x_global) - f_default.f_global) <= f_default.epsilon, \
            f'{f.__name__} failed to have x_global result in f_global'
