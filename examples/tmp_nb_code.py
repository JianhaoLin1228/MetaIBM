"""Presentation-only helpers for example.ipynb (styling, display, formatting, plotting).

This module is almost entirely presentation code: functions work on matplotlib / pandas / plain Python
values the notebook passes in, so MetaIBM calls (meta_show_*, meta_distribution_*, attribute reads, ...)
mostly stay in the notebook cells where they document the API rather than the plumbing. The one exception
is plot_selection_fitness_curve(), which sweeps metaibm.habitat.survival_rate to draw the niche curve -- a
plotting detail not worth keeping in the tutorial cell.

Import in the notebook as:  import tmp_nb_code as tmp
Typical use:
    with tmp.square_patch_format(sub_row, sub_col, fmt='g', grid=True):
        meta_obj.meta_show_species_distribution(...)          # <- MetaIBM call lives in the notebook
    tmp.show_saved('species_distribution.jpg', 'my caption')
"""
import os, time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display, clear_output

# Re-exported so the notebook can pull the whole stack in one line: `from tmp_nb_code import *`.
# (The tmp.* helper functions stay reachable via the `import tmp_nb_code as tmp` alias.)
__all__ = ['np', 'pd', 'plt', 'sns', 'time', 'os', 'Image', 'display']


# --------------------------------------------------------------------------- #
# Plot styling + display
# --------------------------------------------------------------------------- #
@contextmanager
def square_patch_format(sub_row, sub_col, *, fmt, annot_size=28, title_size=56, grid=False, grid_color='gray', grid_lw=1.5):
    """Temporarily restyle MetaIBM's built-in patch plotters (no source change): square cells,
    readable title / cell-value fonts, and (optionally) cell borders even on empty cells. On exit it
    also closes the blank figure the plotter leaves open, then restores the originals.
    `sub_row`/`sub_col` is the subplot grid the map is drawn into; `fmt` is the seaborn annotation
    format ('.4f' for env values, 'g' for integer species ids)."""
    orig_figure, orig_heatmap, orig_set_title = plt.figure, sns.heatmap, Axes.set_title
    plt.figure = lambda *a, **k: orig_figure(*a, **{**k, 'figsize': (20 * sub_col, 20 * sub_row)})  # square cells, no whitespace
    def heatmap(*a, **k):
        ax = orig_heatmap(*a, **{**k, 'square': True, 'annot': True, 'fmt': fmt, 'annot_kws': {'size': annot_size}})
        if grid:  # seaborn skips borders on empty (NaN) cells, so overlay the grid ourselves
            nrows, ncols = np.asarray(k.get('data', a[0] if a else None)).shape
            ax.set_xticks(np.arange(ncols + 1), minor=True)
            ax.set_yticks(np.arange(nrows + 1), minor=True)
            ax.grid(which='minor', color=grid_color, linewidth=grid_lw)
            ax.tick_params(which='minor', length=0)  # gridlines only, no minor tick marks
        return ax
    sns.heatmap = heatmap
    Axes.set_title = lambda self, *a, **k: orig_set_title(self, *a, **{**k, 'fontsize': title_size})
    try:
        yield
    finally:
        plt.figure, sns.heatmap, Axes.set_title = orig_figure, orig_heatmap, orig_set_title  # restore the originals
        plt.close('all')  # the plotter leaves a blank figure open; close it so it isn't rendered


def show_saved(file_name, caption=None):
    """Display an image file the notebook just saved (optionally print a caption above it)."""
    if caption:
        print(caption)
    display(Image(file_name))


@contextmanager
def map_style(kind='value', sub_col=1):
    """Notebook square-cell styling for a MetaIBM meta_show_* map, so the notebook cell shows only the
    MetaIBM call itself -- all the styling knobs live here. kind='species' formats integer species ids
    ('g', larger font); kind='value' formats the 4-decimal phenotype / env values. sub_col is the number
    of patches drawn side by side (1 for the single-patch mainland, 2 for the two islands). Thin wrapper
    over square_patch_format that fixes the single-row (sub_row=1) layout."""
    fmt, annot_size = ('g', 28) if kind == 'species' else ('.4f', 20)
    with square_patch_format(sub_row=1, sub_col=sub_col, fmt=fmt, grid=True, annot_size=annot_size):
        yield


def show_saved_montage(specs, nrows, ncols, figsize=None, caption=None, crop=True, wspace=0.02, hspace=0.12, title_size=11, legend=False, legend_at=None, legend_size=None):
    """Arrange several already-saved image files into one nrows x ncols figure. `specs` is a list of
    (loc, file_name, title): loc is the 1-based subplot position in matplotlib order (left->right,
    top->bottom); any position not listed is left blank (e.g. give locs 1,2,3,5,6 to skip cell 4).
    crop=True trims each image's white border (MetaIBM's meta_show_* saves with wide margins) and
    small wspace/hspace pack the panels tightly.
    legend=False titles each panel above its image. legend=True instead leaves the images bare and
    writes all the titles as one text block in a blank grid cell (default the first unused cell -- e.g.
    the bottom-left cell 4 when locs are 1,2,3,5,6), two lines per panel: 'rowR, colC:' then the title.
    legend_at overrides which blank cell holds the block. Purely presentation -- MetaIBM saved the images."""
    import matplotlib.image as mpimg
    if caption:
        print(caption)
    def load(file_name):
        img = mpimg.imread(file_name)
        if not crop:
            return img
        arr = img[..., :3].astype(float)          # drop alpha if any
        arr = arr * 255 if arr.max() <= 1 else arr  # PNGs are 0-1, JPGs are 0-255
        nonwhite = np.any(arr < 245, axis=-1)       # any non-near-white pixel is content
        if crop == 'bbox':
            # Trim only the outer white margin, keeping everything inside (e.g. the patch1/patch2 titles
            # and the gap between two side-by-side patches) -- use for multi-patch maps.
            rows, cols = np.where(nonwhite.any(axis=1))[0], np.where(nonwhite.any(axis=0))[0]
        else:
            # Dense mode (default): keep only rows/cols that are *mostly* non-white, dropping the sparse
            # 'patchN' title text and axis labels but keeping the solid heatmap grid.
            rows = np.where(nonwhite.mean(axis=1) > 0.5)[0]
            cols = np.where(nonwhite.mean(axis=0) > 0.5)[0]
            if not (len(rows) and len(cols)):        # fallback: crop to any non-white content
                rows, cols = np.where(nonwhite.any(axis=1))[0], np.where(nonwhite.any(axis=0))[0]
        return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1] if len(rows) and len(cols) else img
    fig = plt.figure(figsize=figsize or (4 * ncols, 4 * nrows))
    legend_lines = []
    for loc, file_name, title in specs:
        ax = fig.add_subplot(nrows, ncols, loc)
        ax.imshow(load(file_name))
        ax.axis('off')
        if legend:
            row, col = (loc - 1) // ncols + 1, (loc - 1) % ncols + 1
            legend_lines.append(f'row{row}, col{col}:\n {title}')
        else:
            ax.set_title(title, fontsize=title_size)
    if legend and legend_lines:                     # drop the title block into a blank cell
        blanks = [loc for loc in range(1, nrows * ncols + 1) if loc not in {s[0] for s in specs}]
        at = legend_at if legend_at is not None else (blanks[0] if blanks else None)
        if at is not None:
            ax = fig.add_subplot(nrows, ncols, at)
            ax.axis('off')
            ax.text(0.0, 0.5, '\n'.join(legend_lines), va='center', ha='left', fontsize=legend_size or title_size, family='monospace', transform=ax.transAxes)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)  # pack panels; no tight_layout (it re-pads)
    plt.show()


def show_mainland_montage(**overrides):
    """One-line montage of the mainland's five saved maps -- species / phenotype-1 / phenotype-2 /
    env-axis-1 / env-axis-2 -- in a labelled 3 x 2 grid (legend in the blank top-right cell). Just calls
    show_saved_montage with the mainland's file list, titles, and the tuned layout; pass overrides (e.g.
    legend_size=24, figsize=(30, 45)) to tweak the presentation without touching the notebook cell."""
    specs = [(1, 'mainland_species_distribution.jpg',  "mainland's species distribution"),
             (3, 'mainland_phenotype_phenotype-1.jpg', "mainland's phenotype-1 distribution"),
             (4, 'mainland_phenotype_phenotype-2.jpg', "mainland's phenotype-2 distribution"),
             (5, 'mainland_env-axis-1.jpg',            "mainland's env-axis-1 distribution"),
             (6, 'mainland_env-axis-2.jpg',            "mainland's env-axis-2 distribution")]
    kwargs = dict(nrows=3, ncols=2, figsize=(24, 36), legend=True, legend_size=20, wspace=0.015, hspace=0.015, caption='The mainland at a glance  (3 x 2; the blank top-right cell lists each panel):')
    show_saved_montage(specs, **{**kwargs, **overrides})


def show_meta_montage(include_env=True, **overrides):
    """One-line montage of meta_obj's two islands: stacked rows -- species / phenotype-1 / phenotype-2, plus
    env-axis-1 / env-axis-2 when include_env=True -- where each row is one meta_show_* figure showing
    patch1 | patch2 side by side. Calls show_saved_montage in an N x 1 layout (bbox crop keeps the patch
    labels); pass include_env=False to drop the two environment rows, or overrides (e.g. figsize=(18, 44))."""
    specs = [(1, 'islands_species_distribution.jpg',      'species distribution'),
             (2, 'islands_phenotype_phenotype-1.jpg',     'phenotype-1 distribution'),
             (3, 'islands_phenotype_phenotype-2.jpg',     'phenotype-2 distribution')]
    if include_env:
        specs += [(4, 'islands_env-axis-1.jpg', 'env-axis-1 distribution'), (5, 'islands_env-axis-2.jpg', 'env-axis-2 distribution')]
    nrows = len(specs)
    kwargs = dict(nrows=nrows, ncols=1, figsize=(14, 6.8 * nrows), crop='bbox', title_size=16, wspace=0.02, hspace=0.10, caption=f'The two islands at a glance  ({nrows} rows: the maps; 2 columns within each row: patch1 | patch2):')
    show_saved_montage(specs, **{**kwargs, **overrides})


# --------------------------------------------------------------------------- #
# Small value formatters (tables, grids, pools, arrays) -- take plain values, return display objects
# --------------------------------------------------------------------------- #
def attr_table(rows, columns=None, mono=False):
    """A small table for display. `rows` is a list of tuples (one per row); by default it is a
    two-column Attributes/Values table. Pass `columns` for custom headers -- e.g. a 3-column
    ['attribute', 'patch1', 'patch2'] table with one row per attribute.
    mono=True renders values left-aligned in monospace (for multi-line genotype dumps)."""
    df = pd.DataFrame(rows, columns=columns if columns is not None else ['Attributes', 'Values'])
    if not mono:
        return df
    pd.set_option('display.max_colwidth', None)
    return (df.style
            .set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left', 'font-family': 'monospace'})
            .set_table_styles([{'selector': 'thead th', 'props': [('text-align', 'center')]}]))  # centre the headers


def count_genders(species_category, gender_names=('female', 'male')):
    """Count individuals per gender in a habitat's species_category dict ({species: {gender: [positions]}}).
    Pure dict work -- pass `habitat.species_category` from the notebook. Returns {gender: count}."""
    return {g: sum(len(sp_genders.get(g, [])) for sp_genders in species_category.values()) for g in gender_names}


def expect_offspring_num(gender_counts, birth_rate):
    """Expected offspring given a {gender: count} dict (from count_genders) and the sexual birth rate:
    birth_rate * min(females, males) -- pair count (limited by the rarer gender) times per-pair births.
    Single-species form; for multiple species sum birth_rate*min(females, males) over each species."""
    return birth_rate * min(gender_counts.get('female', 0), gender_counts.get('male', 0))


def habitat_state(h):
    """One habitat's occupancy + pool sizes as a pd.Series of 'a / b' strings, for side-by-side display
    (e.g. before vs after germination). Reads the habitat's counts (empty/occupied sites, offspring/immigrant pools)."""
    return pd.Series({'empty / occupied sites':    f'{len(h.empty_site_pos_ls)} / {h.indi_num}',
                      'offspring / immigrant pool': f'{len(h.offspring_pool)} / {len(h.immigrant_pool)}'})


def pool_sizes(hab1, hab2, names=('patch1', 'patch2')):
    """Offspring/immigrant pool sizes for two habitats as a dict {'<name> offspring_pool': n, '<name> immigrant_pool': n, ...},
    e.g. for a before/after clear-up table. Reads each habitat's offspring_pool and immigrant_pool."""
    sizes = {}
    for name, h in zip(names, (hab1, hab2)):
        sizes[f'{name} offspring_pool'] = len(h.offspring_pool)
        sizes[f'{name} immigrant_pool'] = len(h.immigrant_pool)
    return sizes


def format_genotype_set(genotype_set):
    """Compactly format an individual's genotype_set dict: each trait's two 0/1 arrays on their own
    aligned line, braces/brackets kept inline. Returns a string."""
    np.set_printoptions(linewidth=200)   # keep each locus array on a single line
    items = list(genotype_set.items())
    lines = []
    for idx, (trait_name, alleles) in enumerate(items):
        prefix = f"{'{' if idx == 0 else ' '}{trait_name!r}: ["
        joined = (',\n' + ' ' * len(prefix)).join(repr(allele) for allele in alleles)
        lines.append(prefix + joined + ']' + ('}' if idx == len(items) - 1 else ','))
    return '\n'.join(lines)


def format_microsite_grid(grid, occupied_label=None):
    """Render a habitat's microsite_individuals (a 2D list of individual-or-None) as an aligned string:
    empty cells show None; occupied cells show each individual's own repr -- the real object with its
    memory address, e.g. <metaibm.individual.individual object at 0x...> (repr avoids individual.__str__,
    which would dump the whole genotype/phenotype). Pass occupied_label to substitute a fixed placeholder
    string for every occupied cell instead. All cells are padded to a common width."""
    tokens = [[(repr(occupied_label) if occupied_label is not None else repr(cell)) if cell is not None else repr(None) for cell in row] for row in grid]
    width = max((len(t) for row in tokens for t in row), default=4)
    rows_str = ['[' + ', '.join(t.ljust(width) for t in row) + ']' for row in tokens]
    return '[' + ',\n '.join(rows_str) + ']'


def format_pool(pool, occupied_label='<individual object>', per_line=5):
    """Render a pool (offspring_pool / immigrant_pool -- a flat list of individuals), `per_line` per line,
    to keep a long list compact. Each element shows `occupied_label`; pass occupied_label=None to show
    every individual's real repr instead (the object with its memory address, e.g. <...individual object at 0x...>)."""
    items = [(repr(occupied_label) if occupied_label is not None else repr(obj)) for obj in pool]
    rows = [', '.join(items[i:i + per_line]) for i in range(0, len(items), per_line)]
    return '[' + ',\n '.join(rows) + ']'


def format_env_array(arr):
    """Render a habitat env grid (numpy array) as a compact, copy-pasteable np.array(...) string."""
    return 'np.array(%s)' % np.array2string(arr, separator=', ', prefix='np.array(', max_line_width=1000)


# --------------------------------------------------------------------------- #
# Time-series + video (operate on the recorded pandas tables -- no MetaIBM objects)
# --------------------------------------------------------------------------- #
def plot_run_timeseries(species, pheno1, pheno2, time_axis, source=0.2, optimum=0.4):
    """Three figures (abundance, mean phenotype-1, mean phenotype-2), each split into patch1 (left) /
    patch2 (right), with `time_axis` on the x-axis. `source` / `optimum` mark the mainland and island means."""
    patch_ids = species.columns.get_level_values(0)   # one patch id per microsite column
    def per_patch(table, pid, how):
        cols = table.loc[:, patch_ids == pid]
        return cols.notna().sum(axis=1) if how == 'abundance' else cols.mean(axis=1)   # abundance = occupied count; mean = pop average
    figures = [('abundance',   'abundance (individuals)', species, 'abundance'),
               ('phenotype-1', 'mean phenotype-1',        pheno1,  'mean'),
               ('phenotype-2', 'mean phenotype-2',        pheno2,  'mean')]
    for key, ylabel, table, how in figures:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, pid in zip(axes, ('patch1', 'patch2')):
            ax.plot(time_axis, per_patch(table, pid, how), color='tab:blue', lw=1.8)
            if how == 'mean':
                ax.axhline(source, color='gray', ls=':', lw=1.2, label='mainland source (%.1f)' % source)
                ax.axhline(optimum, color='crimson', ls='--', lw=1.2, label='island optimum (%.1f)' % optimum)
                ax.legend(loc='best', fontsize=8)
            ax.set(title=pid, xlabel='time step', ylabel=ylabel)
        fig.suptitle(ylabel + ' over time', fontsize=13)
        plt.tight_layout()
        file_name = 'run_%s_over_time.jpg' % key
        plt.savefig(file_name, dpi=110)
        plt.close('all')
        display(Image(file_name))


def render_run_video(species, pheno1, pheno2, file_name='run_maps.gif', fps=100, hab_len=10):
    """Rebuild each recorded snapshot into six maps (species / phenotype-1 / phenotype-2, per patch)
    and animate one frame per recorded time-step. Empty microsites (NaN) show as white.
    Returns (file_name, n_frames)."""
    patch_ids = species.columns.get_level_values(0)
    patches = ('patch1', 'patch2')
    metrics = [('species', species, 'tab10', 1, 5), ('phenotype-1', pheno1, 'viridis', 0.0, 0.8), ('phenotype-2', pheno2, 'viridis', 0.0, 0.8)]
    frame_data = {(m[0], pid): m[1].loc[:, patch_ids == pid].to_numpy() for m in metrics for pid in patches}
    steps = species.index.str.replace('time_step', '', regex=False).astype(int)
    n_frames = len(species)

    fig, axes = plt.subplots(3, 2, figsize=(6, 8), constrained_layout=True)
    ims = {}
    for r, (mname, _, cmap, vmin, vmax) in enumerate(metrics):
        cmap_obj = plt.get_cmap(cmap).copy(); cmap_obj.set_bad('white')   # empty microsite -> white
        for col, pid in enumerate(patches):
            ax = axes[r][col]
            ims[(mname, pid)] = ax.imshow(frame_data[(mname, pid)][0].reshape(hab_len, hab_len), cmap=cmap_obj, vmin=vmin, vmax=vmax)
            ax.set_title(f'{mname} · {pid}', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    sup = fig.suptitle('', fontsize=12)

    def update(t):
        for key, im in ims.items():
            im.set_data(frame_data[key][t].reshape(hab_len, hab_len))
        sup.set_text('time step %d' % steps[t])
        return list(ims.values()) + [sup]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(file_name, writer=PillowWriter(fps=fps))
    plt.close('all')
    return file_name, n_frames


# --------------------------------------------------------------------------- #
# Selection curve (the one helper that calls a MetaIBM method: hab.survival_rate)
# --------------------------------------------------------------------------- #
_fitness_curve_envs = []   # environments accumulated across successive plot_selection_fitness_curve() calls

def plot_selection_fitness_curve(survival_rate_func, env_val, base_dead_rate, fitness_wid, *, source=0.2, file_name='selection_fitness_curve.jpg'):
    """Niche-Gaussian survival vs phenotype, drawn one environment per call. `survival_rate_func` is
    metaibm.habitat.survival_rate (its `self` is unused, so we pass None); `base_dead_rate`/`fitness_wid`
    are its d/w. Successive calls accumulate onto a single figure, so the cell's output stays one combined
    plot: the `source` env (mainland, called first) starts a fresh figure and carries the 1-d ceiling and
    sp1 optimum marker; later envs (islands) are overlaid. Saves to `file_name` and displays it."""
    global _fitness_curve_envs
    if abs(env_val - source) < 1e-12: _fitness_curve_envs = []   # the mainland call starts a fresh figure
    _fitness_curve_envs.append(env_val)

    survival = lambda phenotype_ls, env_val_ls: survival_rate_func(None, d=base_dead_rate, phenotype_ls=phenotype_ls, env_val_ls=env_val_ls, w=fitness_wid, method='niche_gaussian')
    phenotypes = np.linspace(0.0, 0.8, 161)
    plt.figure(figsize=(8, 5))
    for env in _fitness_curve_envs:
        is_source = abs(env - source) < 1e-12
        colour, label = ('tab:blue', f'mainland env ~ {source:g} (sp1 source optimum)') if is_source else ('tab:orange', f'island env ~ {env:g} (both islands)')
        curve = [survival([p, p], [env, env]) for p in phenotypes]
        plt.plot(phenotypes, curve, color=colour, lw=2.5, label=label)
        plt.axvline(env, color=colour, ls=':', lw=1.2)   # mark each environment optimum
    ceiling = survival([source, source], [source, source])   # sp1 at its source: perfect match -> 1 - d
    plt.axhline(ceiling, color='gray', ls='--', lw=1, label=f'ceiling 1 - d = {ceiling:g}')
    plt.scatter([source], [ceiling], color='tab:blue', zorder=5, s=70, label=f'sp1 at its {source:g} source optimum (survival = {ceiling:g})')
    plt.title('Niche-Gaussian survival vs phenotype')
    plt.xlabel('phenotype (both traits)')
    plt.ylabel('survival probability')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_name, dpi=110)
    plt.close('all')
    clear_output(wait=True)   # replace the previous call's image so the cell shows one combined figure
    display(Image(file_name))
