#!/usr/bin/env python3
"""Plot persisted gravity results; uses only system matplotlib and stdlib."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out = Path(sys.argv[1])
    report = json.loads((out/'results.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, name in zip(axes.flat, ('compact', 'extended', 'grasp', 'lift')):
        for run in report['runs']:
            if (run['pose'] != name or run['gain'] != 'low' or run['ff'] != 'mujoco_bias'
                or run['mass_scale'] != 1 or run['com_shift_m'] != 0 or not run['name'].endswith('_d0')):
                continue
            with (out/run['csv']).open() as stream:
                reader=csv.reader(stream); next(reader)
                rows=[[float(v) for v in row] for row in reader]
            ax.plot([r[0] for r in rows], [max(abs(v) for v in r[1:7]) for r in rows],
                    label=f"bias scale {run['scale']:g}")
        ax.set_title(name); ax.set_ylabel('max joint error (deg)')
        ax.grid(alpha=.3); ax.legend()
    for ax in axes[-1]: ax.set_xlabel('time (s)')
    fig.tight_layout(); fig.savefig(out/'hold_error.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
