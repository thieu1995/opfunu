import numpy as np
from opfunu import get_all_cec_functions


def test_whenNdimNone_thenDefaultNdimUsed():
    allFunctions = get_all_cec_functions()
    for f in allFunctions:
        f_default = f()
        assert f_default.ndim == f_default.dim_default, f'{f.__name__} failed to have ndim == dim_default'


def test_whenEvaulateDefaultNdim_thenHasResult():
    known_failing = []
    all_functions = [x for x in get_all_cec_functions() if x.__name__ not in known_failing]
    failing = []
    for f in all_functions:
        f_default = f()
        x = np.random.rand(f_default.dim_default)
        try:
            f_default.evaluate(x)
        except x:
            print(f'{f_default.__name__}:{f_default.dim_default}:{x}')
            failing.append(f.__name__)

    assert len(failing) == 0, f'{failing} failed to have x_global result in f_global'


def test_whenEvaulateWith_x_global_then_f_global():
    # The following are broken or have incorrect or unknown correct , values.
    known_failing = ['F72005', 'F72008', 'F102014', 'F112014',
                     'F132014', 'F142014', 'F172014', 'F182014', 'F192014', 'F202014', 'F212014',
                     'F222014', 'F232014', 'F242014', 'F252014', 'F262014', 'F272014', 'F282014',
                     'F292014', 'F302014', 'F42015', 'F62015', 'F72015', 'F102015', 'F112015',
                     'F122015', 'F132015', 'F142015', 'F152015', 'F82017', 'F92017', 'F102017',
                     'F112017', 'F122017', 'F142017', 'F152017', 'F162017', 'F172017', 'F182017',
                     'F192017', 'F202017', 'F212017', 'F222017', 'F232017', 'F242017', 'F252017',
                     'F262017', 'F272017', 'F282017', 'F292017', 'F12019', 'F22019', 'F32019',
                     'F72019', 'F92019', 'F52020', 'F62020', 'F72020', 'F82020', 'F92020',
                     'F102020', 'F152014', 'F42020', 'F82015']
    all_functions = [x for x in get_all_cec_functions() if x.__name__ not in known_failing]
    failing = []
    for f in all_functions:
        f_default = f()
        x_global = f_default.x_global
        if abs(f_default.evaluate(x_global) - f_default.f_global) >= f_default.epsilon:
            failing.append(f.__name__)
    assert len(failing) == 0, f'{failing} failed to have x_global result in f_global'


