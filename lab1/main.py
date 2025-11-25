import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

from lab1.utils import *
from lab1.sobel import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--outdir', default='out')
    args = parser.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img = load_image(str(inp))

    funcs = [
        ('opencv', sobel_opencv),
        ('native_loops', sobel_native_loops),
        ('native_vectorized', sobel_native_vectorized),
    ]

    results = []

    for name, func in funcs:
        mean_t, std_t, all_ts = benchmark(func, img, runs=args.runs)
        out = func(img)
        save_uint8_image(out, str(outdir / f'output_{name}.png'))
        results.append((name, mean_t, std_t))

    csv_path = outdir / 'benchmark_results.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name', 'mean_s', 'std_s'])
        for r in results:
            w.writerow(r)

    names = [r[0] for r in results]
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(8,4))
    plt.bar(names, means, yerr=stds)
    plt.ylabel('Time (s)')
    plt.title('Sobel benchmark')
    plt.tight_layout()
    plt.savefig(outdir / 'benchmark_times.png')


if __name__ == '__main__':
    main()
