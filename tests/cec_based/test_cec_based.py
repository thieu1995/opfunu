import numpy as np

from opfunu import get_all_cec_functions


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_cec_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f'{f.__name__} failed to have ndim == dim_default'

def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct , values.
    known_failing = ['F102017', 'F102020', 'F102021', 'F102022', 'F112015', 'F112022',
                     'F12019', 'F122015', 'F122017', 'F122022', 'F132014', 'F132015',
                     'F142014', 'F142015', 'F142017', 'F152015', 'F152017', 'F162017',
                     'F172017', 'F182014', 'F182017', 'F192014', 'F192017', 'F202014',
                     'F202017', 'F212013', 'F212014', 'F212017', 'F22019', 'F222013',
                     'F222014', 'F222017', 'F232013', 'F232014', 'F232017', 'F242013',
                     'F242014', 'F242017', 'F252013', 'F252014', 'F252017', 'F262017',
                     'F272013', 'F272014', 'F272017', 'F282013', 'F282014', 'F282017',
                     'F292014', 'F292017', 'F302014', 'F32019', 'F52022', 'F62015',
                     'F62020', 'F62021', 'F62022', 'F72005', 'F72008', 'F72015',
                     'F72020', 'F72021', 'F72022', 'F82017', 'F82020', 'F82021',
                     'F82022', 'F92019', 'F92020', 'F92021', 'F92022']
    allFunctions = [x for x in get_all_cec_functions() if x.__name__ not in known_failing]
    for f in allFunctions:
        f_default = f()
        x_global = f_default.x_global
        assert abs(f_default.evaluate(x_global) - f_default.f_global) <= f_default.epsilon, \
            f'{f.__name__} failed to have x_global result in f_global'
