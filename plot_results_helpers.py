import pandas as pd
import numpy as np

def build_per_epoch_rvs(result):
    """
    Build [Epoch, RV1, RV1_uncertainty, RV2, RV2_uncertainty].
    If ratio param => RV1 = - ratio * RV2. Then do star assignment by majority.
    """
    all_params = list(result.params.keys())
    has_ratio = ('ratio' in all_params)
    epoch_nums = set()

    for p in all_params:
        if p.startswith('rv1_epoch'):
            e = int(p.replace('rv1_epoch',''))
            epoch_nums.add(e)
        elif p.startswith('rv2_epoch'):
            e = int(p.replace('rv2_epoch',''))
            epoch_nums.add(e)

    ratio_val = result.params['ratio'].value if has_ratio else 0.0
    ratio_err = result.params['ratio'].stderr if (has_ratio and result.params['ratio'].stderr) else 0.0

    rows = []
    for e in sorted(epoch_nums):
        if has_ratio:
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv2_err = result.params[f'rv2_epoch{e}'].stderr if result.params[f'rv2_epoch{e}'].stderr else 0.0
            rv1_val = - ratio_val * rv2_val
            rv1_err = np.sqrt((rv2_val**2)*(ratio_err**2) + (ratio_val**2)*(rv2_err**2))
        else:
            rv1_val = result.params[f'rv1_epoch{e}'].value
            rv1_err = result.params[f'rv1_epoch{e}'].stderr if result.params[f'rv1_epoch{e}'].stderr else 0.0
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv2_err = result.params[f'rv2_epoch{e}'].stderr if result.params[f'rv2_epoch{e}'].stderr else 0.0

        rows.append({
            'Epoch': e,
            'RV1': rv1_val,
            'RV1_uncertainty': rv1_err,
            'RV2': rv2_val,
            'RV2_uncertainty': rv2_err
        })

    # Star assignment
    count_star2_bigger = sum(abs(r['RV2'])>abs(r['RV1']) for r in rows)
    if count_star2_bigger < len(rows)/2:
        # Swap them
        for r in rows:
            old1, old1err = r['RV1'], r['RV1_uncertainty']
            r['RV1'] = r['RV2']
            r['RV1_uncertainty'] = r['RV2_uncertainty']
            r['RV2'] = old1
            r['RV2_uncertainty'] = old1err

    return pd.DataFrame(rows)
