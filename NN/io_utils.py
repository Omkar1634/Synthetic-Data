"""
Deep Albedo — I/O Utilities

Saving inference outputs (parameter maps, recovered images, metadata)
and generating text/JSON reports.
Used by inference.py and latent_space_validation.py.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# from utils import plotting


# ── Inference outputs ─────────────────────────────────────────────────────────

def save_results(image_path, original, recovered, parameter_maps,
                 dimensions, encode_time, decode_time, output_dir):
    """
    Save all outputs for one processed image:

        output_dir/
        ├── visualizations/          analysis plots
        ├── recovered/               reconstructed RGB
        ├── parameter_maps/Cm|Ch|Bm|Bh|T/   greyscale + coloured maps
        └── data/                    .npy arrays + metadata JSON

    Args:
        image_path:     original image path (used for naming)
        original:       (H, W, 3) float32 [0,1] input image
        recovered:      (H, W, 3) float32 [0,1] reconstructed image
        parameter_maps: (H*W, 5) float32 encoded parameters
        dimensions:     (H, W) tuple
        encode_time:    seconds
        decode_time:    seconds
        output_dir:     root directory to write into
    """
    out   = Path(output_dir)
    name  = Path(image_path).stem
    pnames = ['Cm', 'Ch', 'Bm', 'Bh', 'T']

    viz_dir   = out / "visualizations"
    rec_dir   = out / "recovered"
    param_dir = out / "parameter_maps"
    data_dir  = out / "data"

    for d in [viz_dir, rec_dir, data_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for p in pnames:
        (param_dir / p).mkdir(parents=True, exist_ok=True)

    # Analysis visualisation
    # plt.style.use("dark_background")
    # plt.rcParams["axes.grid"] = False
    # plotting.PLOT_TEX_MAPS(
    #     recovered, parameter_maps,
    #     title=f"Analysis: {name}",
    #     save=True,
    #     text_below=(f"Encode: {encode_time:.4f}s | "
    #                 f"Decode: {decode_time:.4f}s | "
    #                 f"Total: {encode_time + decode_time:.4f}s"),
    # )
    temp = Path(f"tex_maps_Analysis: {name}.png")
    if temp.exists():
        shutil.move(str(temp), str(viz_dir / f"{name}_analysis.png"))
    plt.close('all')

    # Recovered image
    cv2.imwrite(
        str(rec_dir / f"{name}_recovered.png"),
        cv2.cvtColor((recovered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # Per-parameter maps
    pm = parameter_maps.reshape(dimensions[0], dimensions[1], 5)
    for i, pname in enumerate(pnames):
        pmap = pm[:, :, i]
        grey = ((pmap - pmap.min()) /
                (pmap.max() - pmap.min() + 1e-8) * 255).astype(np.uint8)
        cv2.imwrite(str(param_dir / pname / f"{name}_{pname}.png"), grey)

        plt.figure(figsize=(6, 6))
        plt.imshow(pmap, cmap='viridis')
        plt.colorbar()
        plt.title(f'{pname} — {name}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(str(param_dir / pname / f"{name}_{pname}_colored.png"),
                    dpi=100, bbox_inches='tight')
        plt.close()

    # Raw arrays
    np.save(data_dir / f"{name}_parameters.npy", parameter_maps)
    np.save(data_dir / f"{name}_original.npy",   original)
    np.save(data_dir / f"{name}_recovered.npy",  recovered)

    # Metadata JSON
    metadata = {
        'image_name':  name,
        'image_path':  str(image_path),
        'dimensions':  dimensions,
        'encode_time': float(encode_time),
        'decode_time': float(decode_time),
        'total_time':  float(encode_time + decode_time),
        'parameters':  {
            pname: {
                'min':  float(pm[:, :, i].min()),
                'max':  float(pm[:, :, i].max()),
                'mean': float(pm[:, :, i].mean()),
            }
            for i, pname in enumerate(pnames)
        },
    }
    with open(data_dir / f"{name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)


def save_summary_report(results, failed, output_dir, input_folder):
    """
    Save a JSON + TXT batch-processing summary.

    Args:
        results:      list of result dicts from process_single_image()
        failed:       list of {'file': ..., 'error': ...} dicts
        output_dir:   directory to write into
        input_folder: original input folder path (for the report header)
    """
    out    = Path(output_dir)
    n_ok   = len(results)
    n_fail = len(failed)
    n_tot  = n_ok + n_fail

    summary = {
        'processing_info': {
            'date':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_folder':  str(input_folder),
            'output_folder': str(output_dir),
            'total_images':  n_tot,
            'successful':    n_ok,
            'failed':        n_fail,
            'success_rate':  f"{n_ok / n_tot * 100:.2f}%" if n_tot else "0%",
        },
        'performance': {
            'avg_encode_time': f"{np.mean([r['encode_time'] for r in results]):.4f}s",
            'avg_decode_time': f"{np.mean([r['decode_time'] for r in results]):.4f}s",
            'avg_total_time':  f"{np.mean([r['total_time']  for r in results]):.4f}s",
            'total_time':      f"{sum(r['total_time'] for r in results):.2f}s",
        },
        'processed_images': [
            {'filename':    Path(r['image_path']).name,
             'encode_time': f"{r['encode_time']:.4f}s",
             'decode_time': f"{r['decode_time']:.4f}s",
             'total_time':  f"{r['total_time']:.4f}s"}
            for r in results
        ],
        'failed_images': failed,
    }

    json_path = out / 'processing_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    pi, perf = summary['processing_info'], summary['performance']
    txt_path = out / 'processing_summary.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\nDEEP ALBEDO — BATCH PROCESSING SUMMARY\n" + "=" * 70 + "\n\n")
        f.write(f"Date:          {pi['date']}\n"
                f"Input folder:  {pi['input_folder']}\n"
                f"Output folder: {pi['output_folder']}\n\n"
                f"Total images:  {pi['total_images']}\n"
                f"Successful:    {pi['successful']}\n"
                f"Failed:        {pi['failed']}\n"
                f"Success rate:  {pi['success_rate']}\n\n"
                f"Avg encode:    {perf['avg_encode_time']}\n"
                f"Avg decode:    {perf['avg_decode_time']}\n"
                f"Avg total:     {perf['avg_total_time']}\n"
                f"Total time:    {perf['total_time']}\n")
        if failed:
            f.write("\nFailed images:\n")
            for fail in failed:
                f.write(f"  - {fail['file']}: {fail['error']}\n")

    print(f"\n✓ Summary → {json_path}  {txt_path}")


# ── Validation report ─────────────────────────────────────────────────────────

def generate_validation_report(stats, corr_matrix, problematic_corrs,
                                image_stats, output_dir):
    """
    Write a human-readable latent-space validation report.

    Args:
        stats:              output of metrics.analyze_parameter_distribution()
        corr_matrix:        (5, 5) numpy array
        problematic_corrs:  output of metrics.check_parameter_correlations()
        image_stats:        dict with total_images, total_pixels, avg_skin_ratio
        output_dir:         directory to write validation_report.txt

    Returns:
        path to the written report file
    """
    param_names = ['Cm', 'Ch', 'Bm', 'Bh', 'T']
    param_desc  = {
        'Cm': 'Melanin Concentration',
        'Ch': 'Haemoglobin Concentration',
        'Bm': 'Melanin Blend',
        'Bh': 'Blood Oxygenation',
        'T':  'Epidermis Thickness (cm)',
    }

    all_in_range      = all(s['within_expected_range'] for s in stats.values())
    boundary_problems = [n for n, s in stats.items()
                         if s['boundary_clustering']['is_problem']]
    high_outliers     = [n for n, s in stats.items() if s['outlier_pct'] > 5]

    path = f'{output_dir}/validation_report.txt'
    with open(path, 'w') as f:
        f.write("=" * 80 + "\nLATENT SPACE RANGE VALIDATION REPORT\n" + "=" * 80 + "\n")
        f.write(f"Generated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Images:         {image_stats['total_images']}\n"
                f"Skin pixels:    {image_stats['total_pixels']}\n"
                f"Avg skin ratio: {image_stats['avg_skin_ratio']:.2%}\n\n")

        f.write("OVERALL ASSESSMENT\n" + "=" * 80 + "\n")
        if all_in_range and not boundary_problems and not high_outliers:
            f.write("✓ All parameters within expected ranges\n"
                    "✓ No significant boundary clustering\n"
                    "✓ Low outlier rate\n"
                    "CONCLUSION: Model produces biologically plausible parameters.\n\n")
        else:
            if not all_in_range:
                f.write("✗ Some parameters outside expected ranges\n")
            if boundary_problems:
                f.write(f"✗ Boundary clustering: {', '.join(boundary_problems)}\n")
            if high_outliers:
                f.write(f"✗ High outlier rate:   {', '.join(high_outliers)}\n")
            f.write("CONCLUSION: Model needs investigation. See details below.\n\n")

        f.write("PARAMETER DETAILS\n" + "=" * 80 + "\n\n")
        for name in param_names:
            s  = stats[name]
            bc = s['boundary_clustering']
            f.write(f"{name} — {param_desc[name]}\n" + "-" * 60 + "\n"
                    f"  Mean / Std:       {s['mean']:.6f} / {s['std']:.6f}\n"
                    f"  Min  / Max:       {s['min']:.6f} / {s['max']:.6f}\n"
                    f"  Median [Q25–Q75]: {s['median']:.6f} [{s['q25']:.6f}–{s['q75']:.6f}]\n"
                    f"  Within range:     {'✓' if s['within_expected_range'] else '✗'}\n"
                    f"  Bio plausibility: {s['biological_plausibility_pct']:.1f}%\n"
                    f"  Outliers:         {s['outlier_pct']:.2f}%\n"
                    f"  Lower boundary:   {bc['lower_boundary_pct']:.1f}%  "
                    f"Upper: {bc['upper_boundary_pct']:.1f}%  "
                    f"Problem: {'✗ YES' if bc['is_problem'] else '✓ NO'}\n\n")

        f.write("CORRELATION MATRIX\n" + "=" * 80 + "\n")
        f.write("     " + "  ".join(f"{n:>6}" for n in param_names) + "\n")
        for i, n in enumerate(param_names):
            f.write(f"{n:>4} " +
                    "  ".join(f"{corr_matrix[i, j]:>6.2f}"
                               for j in range(len(param_names))) + "\n")
        f.write("\n")
        if problematic_corrs:
            for c in problematic_corrs:
                f.write(f"⚠ {c['params'][0]} vs {c['params'][1]}: "
                        f"{c['correlation']:.2f} — {c['issue']}\n")
        else:
            f.write("✓ No problematic correlations.\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Report → {path}")
    return path
