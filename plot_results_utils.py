# plot_results_utils.py
import pandas as pd

def build_per_epoch_rvs(result):
    """
    If ratio in params => rv1 = -ratio*rv2
    Then we do star assignment if needed
    """
    pnames = list(result.params.keys())
    has_ratio = 'ratio' in pnames

    # collect epochs
    epoch_nums = set()
    for p in pnames:
        if p.startswith('rv1_epoch'):
            e = int(p.replace('rv1_epoch',''))
            epoch_nums.add(e)
        elif p.startswith('rv2_epoch'):
            e = int(p.replace('rv2_epoch',''))
            epoch_nums.add(e)

    ratio_val = result.params['ratio'].value if has_ratio else None
    ratio_err = result.params['ratio'].stderr if (has_ratio and result.params['ratio'].stderr) else 0.0

    rows = []
    for e in sorted(epoch_nums):
        if has_ratio:
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv2_err = result.params[f'rv2_epoch{e}'].stderr or 0.0
            rv1_val = - ratio_val * rv2_val
            # propagate error
            rv1_err = (abs(rv2_val)*ratio_err + abs(ratio_val)*rv2_err)
        else:
            rv1_val = result.params[f'rv1_epoch{e}'].value
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv1_err = result.params[f'rv1_epoch{e}'].stderr or 0.0
            rv2_err = result.params[f'rv2_epoch{e}'].stderr or 0.0

        rows.append({
            'Epoch': e,
            'RV1': rv1_val,
            'RV1_uncertainty': rv1_err,
            'RV2': rv2_val,
            'RV2_uncertainty': rv2_err
        })

    # decide star assignment by majority if you want
    # omitted for brevity

    df = pd.DataFrame(rows)
    return df
