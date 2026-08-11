"""Presentation-only helpers for example2.ipynb (reading runs back, display, formatting, plotting).

Everything here works on plain Python / pandas / matplotlib values that the notebook passes in, so the
MetaIBM calls themselves (the eco-evo loop, meta_distribution_*, get_meta_*, attribute reads) stay visible
in the notebook cells where they document the API rather than the plumbing. The one exception is
plot_fitness_curves(), which sweeps metaibm.habitat.survival_rate to draw the two species' niche curves.

A few generic table/display helpers are shared with the other notebook's helper module (tmp_nb_code.py)
rather than duplicated; everything specific to this model lives here.

Import in the notebook as:  import tmp_nb_code2 as tmp
Typical use:
    run = tmp.load_run(goal_path, species_file, phenotype_file)   # the two recorded csv.gz tables
    tmp.plot_tracking_facets({...}, env_curve=..., steps=run['steps'])
"""
import gzip, os, sys, time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # examples/, where the other notebook's helper module lives -- appended, not inserted, so this folder's own bootstrap_metaibm.py still wins over the one next to example.ipynb
from tmp_nb_code import attr_table

# Re-exported so the notebook can pull the whole stack in one line: `from tmp_nb_code2 import *`.
# (The tmp.* helper functions stay reachable via the `import tmp_nb_code2 as tmp` alias.)
__all__ = ['np', 'pd', 'plt', 'sns', 'time', 'os', 'Image', 'display']

SP_COLOURS = {1: 'tab:blue', 2: 'tab:red'}          # sp1 = cold-adapted (0.2), sp2 = warm-adapted (0.8)
RUN_COLOURS = ['tab:grey', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown', 'tab:cyan']


def run_colour(label, labels):
    """A stable colour per run label, assigned in the order the labels appear."""
    return RUN_COLOURS[list(labels).index(label) % len(RUN_COLOURS)]


# --------------------------------------------------------------------------- #
# The changing environment (a deterministic step function -- the schedule item's gating fields)
# --------------------------------------------------------------------------- #
def env_trajectory(steps, init_env, delta, start, end, interval):
    """The mean environment the metacommunity experiences at each step in `steps`, given a
    meta_offset_environmental_values schedule item gated by {'start', 'end', 'interval'}. Reproduces
    simulator.should_run: the offset fires at start, start+interval, ... up to and including end.
    Returns a numpy array aligned with `steps`."""
    fire_at = np.arange(start, end + 1, interval)
    # the offset fires *before* that step's germination, so step t already lives in the shifted environment
    return np.array([init_env + delta * (fire_at <= t).sum() for t in steps])


def env_change_table(init_env, delta, start, end, interval):
    """The same trajectory as a small table (one row per environmental-change event), for display."""
    fire_at = np.arange(start, end + 1, interval)
    return pd.DataFrame({'event': np.arange(1, len(fire_at) + 1), 'time_step': fire_at, 'mean environment after the change': [round(init_env + delta * (i + 1), 3) for i in range(len(fire_at))]})


# --------------------------------------------------------------------------- #
# Niche / fitness curves (the one helper that calls a MetaIBM method: habitat.survival_rate)
# --------------------------------------------------------------------------- #
def plot_fitness_curves(survival_rate_func, base_dead_rate, fitness_wid, species_phenotypes, environments, file_name='example2_fitness_curves.jpg'):
    """Niche-Gaussian survival vs microsite environment, one curve per species phenotype.
    `survival_rate_func` is metaibm.habitat.survival_rate (its `self` is unused, so we pass None);
    `base_dead_rate` / `fitness_wid` are its d / w. Vertical lines mark the environments the
    metacommunity passes through as it warms. Saves to `file_name` and displays it."""
    survival = lambda pheno, env: survival_rate_func(None, d=base_dead_rate, phenotype_ls=[pheno], env_val_ls=[env], w=fitness_wid, method='niche_gaussian')
    env_axis = np.linspace(0.0, 1.0, 201)

    plt.figure(figsize=(9, 5))
    for i, pheno in enumerate(species_phenotypes):
        plt.plot(env_axis, [survival(pheno, e) for e in env_axis], color=SP_COLOURS[i + 1], lw=2.5, label='sp%d  (phenotype %.1f)' % (i + 1, pheno))
    for env in environments:
        plt.axvline(env, color='gray', ls=':', lw=1.0)
        plt.text(env, 1.01, '%.1f' % env, ha='center', va='bottom', fontsize=8, color='gray')
    plt.axhline(1 - base_dead_rate, color='k', ls='--', lw=1, label='ceiling  1 - d = %.1f' % (1 - base_dead_rate))
    plt.title('Niche-Gaussian survival of the two species along the environmental axis\n(dotted lines: the environments the landscape passes through as it warms)')
    plt.xlabel('microsite environment value')
    plt.ylabel('survival probability per time step')
    plt.ylim(0, 1.05)
    plt.legend(loc='center left')
    plt.tight_layout()
    plt.savefig(file_name, dpi=110)
    plt.close('all')
    display(Image(file_name))


def survival_table(survival_rate_func, base_dead_rate, fitness_wid, species_phenotypes, environments):
    """The same information as a table: survival probability of each species in each environment."""
    survival = lambda pheno, env: survival_rate_func(None, d=base_dead_rate, phenotype_ls=[pheno], env_val_ls=[env], w=fitness_wid, method='niche_gaussian')
    rows = [['sp%d (phenotype %.1f)' % (i + 1, p)] + ['%.3f' % survival(p, e) for e in environments] for i, p in enumerate(species_phenotypes)]
    return attr_table(rows, columns=['species'] + ['env = %.1f' % e for e in environments])


# --------------------------------------------------------------------------- #
# Reading the recorded run back (the two csv.gz tables the recorders wrote)
# --------------------------------------------------------------------------- #
def run_is_complete(goal_path, file_names, expected_rows):
    """True when every file of `file_names` exists under `goal_path` and holds at least `expected_rows`
    lines. The recorder appends one line per snapshot, so a run that was interrupted leaves a short file
    rather than no file -- existence alone is not enough to decide that a run can be reused instead of
    re-simulated. A restarted run can leave more lines than expected (see _read_snapshots), hence '>='."""
    for name in file_names:
        path = os.path.join(goal_path, name)
        if not os.path.exists(path): return False
        with gzip.open(path, 'rt') as fh:
            if sum(1 for _ in fh) < expected_rows: return False
    return True


def final_occupancy(goal_path, species_file):
    """Occupied microsites in the last recorded snapshot, read off the last line of the .csv.gz. This is
    what stands in for metacommunity.get_meta_individual_num() when a run was reused from disk and so has
    no live metacommunity object -- it counts the last *recorded* step, not the last simulated one."""
    with gzip.open(os.path.join(goal_path, species_file), 'rt') as fh: last_line = fh.readlines()[-1]
    return sum(1 for value in last_line.rstrip('\n').split(',')[1:] if value.strip() != '')


def _read_snapshots(file_name):
    """Read one recorded table, tolerating a run that was interrupted and restarted.

    save_snapshot appends each snapshot as a new gzip member, so if a previous grid was interrupted and
    its worker processes were still writing when the grid was launched again, both writers append into
    the same file: the reference row and time_step0 land first, then the two streams interleave and the
    tail of the old run shows up as duplicated, out-of-order rows. Every run re-seeds from `random_seed`
    before it starts, so the duplicated snapshots are byte-identical -- the data is fine, it just has to
    be de-duplicated and put back in time order before anything positional (run_summary, the videos, the
    stage samples) can use it. Keeping the last copy of each label also means a deliberate re-run of one
    parameter combination wins over whatever was in the file before it."""
    table = pd.read_csv(file_name, header=[0, 1, 2], index_col=0)
    table = table[~table.index.duplicated(keep='last')]
    reference = table.index[~table.index.str.startswith('time_step')]
    snapshots = table.index[table.index.str.startswith('time_step')]
    in_time_order = list(reference) + sorted(snapshots, key=lambda label: int(label.replace('time_step', '')))
    return table.loc[in_time_order]


def load_run(goal_path, species_file, phenotype_file):
    """Read the two recorded tables of one scenario and split off their primed header rows.
    Returns a dict with:
      'species'   -- (time x microsite) species ids (1 / 2, NaN = empty microsite)
      'phenotype' -- (time x microsite) phenotype values (NaN = empty microsite)
      'optimum'   -- the primed 'optimun_sp_id_values' row (best species per microsite at t=0)
      'env0'      -- the primed 'environment' row (the initial environment per microsite)
      'steps'     -- the recorded time steps as an int array
      'patch_ids' -- the patch id of every microsite column"""
    species = _read_snapshots(os.path.join(goal_path, species_file))
    phenotype = _read_snapshots(os.path.join(goal_path, phenotype_file))
    optimum, env0 = species.loc['optimun_sp_id_values'], phenotype.loc['environment']
    species, phenotype = species.drop(index=['optimun_sp_id_values']), phenotype.drop(index=['environment'])
    steps = species.index.str.replace('time_step', '', regex=False).astype(int).to_numpy()
    return {'species': species, 'phenotype': phenotype, 'optimum': optimum, 'env0': env0, 'steps': steps, 'patch_ids': species.columns.get_level_values(0)}


def run_summary(run):
    """Per-time-step summary series of one scenario: occupancy, the two species' abundances, and each
    species' mean phenotype (NaN when that species is absent). Returns a DataFrame indexed by time step."""
    species, phenotype = run['species'], run['phenotype']
    if not (species.index.equals(phenotype.index) and species.columns.equals(phenotype.columns)):
        # The two tables are recorded by separate appends, so an interrupted run can leave them different
        # lengths -- `phenotype.where(species == ...)` then dies with "putmask: mask and data must be the
        # same size" several frames deep in pandas. Line them up on the species table instead: _read_snapshots
        # already de-duplicates both, so this only ever fires on tables loaded some other way.
        phenotype = phenotype.reindex(index=species.index, columns=species.columns)
    return pd.DataFrame({'occupancy': species.notna().sum(axis=1).to_numpy(),
                         'sp1': (species == 1).sum(axis=1).to_numpy(),
                         'sp2': (species == 2).sum(axis=1).to_numpy(),
                         'sp1 mean phenotype': phenotype.where(species == 1).mean(axis=1).to_numpy(),
                         'sp2 mean phenotype': phenotype.where(species == 2).mean(axis=1).to_numpy()}, index=run['steps'])


def _patch_order(patch_ids):
    """The patch names in numeric order (patch1, patch2, ... patch16), not the lexical order of a set."""
    return sorted(set(patch_ids), key=lambda p: int(p.replace('patch', '')))


# --------------------------------------------------------------------------- #
# Scenario comparison figures
# --------------------------------------------------------------------------- #
def plot_tracking_facets(summaries_by_facet, env_curve, steps, change_window=None, title='', column='sp1 mean phenotype', ylabel='mean phenotype of sp1', file_name='example2_scenarios.jpg'):
    """One panel per facet, the runs inside a facet overlaid, every panel on shared axes so the whole
    parameter grid is one figure: read the lines *within* a panel for the effect of the overlaid factor,
    and one colour *across* panels for the effect of the facet factor. `summaries_by_facet` is
    {facet label: {run label: run_summary DataFrame}}; each facet is expected to carry the same run
    labels, which is what keeps a run the same colour in every panel. The dotted line is `env_curve`,
    the moving optimum. Saves to `file_name` and displays it."""
    facets = list(summaries_by_facet)
    run_labels = list(next(iter(summaries_by_facet.values())))
    fig, axes = plt.subplots(1, len(facets), figsize=(4.7 * len(facets), 4.2), sharex=True, sharey=True, squeeze=False)
    for ax, facet in zip(axes[0], facets):
        for label, summary in summaries_by_facet[facet].items():
            ax.plot(summary.index, summary[column], color=run_colour(label, run_labels), lw=1.8, label=label)
        ax.plot(steps, env_curve, color='k', lw=1.6, ls=':', label='mean environment (the moving optimum)')
        if change_window is not None:
            ax.axvspan(change_window[0], change_window[1], color='orange', alpha=0.10)
        ax.set(title=facet, xlabel='time step')
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    plt.savefig(file_name, dpi=110, bbox_inches='tight')
    plt.close('all')
    display(Image(file_name))


# --------------------------------------------------------------------------- #
# Per-patch species frequency, and the alternative-stable-states test
# --------------------------------------------------------------------------- #
def stage_samples(steps, env_curve, digits=3):
    """The last recorded time step spent at each successive environmental level -- i.e. one settled sample
    per climate stage, taken just before the next change. Returns [(environment, time_step), ...] in the
    order the stages occur (a dict keeps insertion order, and each later step overwrites the earlier one)."""
    samples = {}
    for step, env in zip(steps, np.round(np.asarray(env_curve), digits)):
        samples[float(env)] = int(step)
    return list(samples.items())


def _snapshot_patch_frequency(snapshot, patch_ids, order, species_id):
    """Frequency of `species_id` inside every patch of one recorded snapshot, NaN where a patch is empty."""
    frequencies = []
    for p in order:
        cells = snapshot[patch_ids == p]
        occupied = cells.notna().sum()
        frequencies.append((cells == species_id).sum() / occupied if occupied else np.nan)
    return frequencies


def stage_frequencies(run, env_curve, species_id=1):
    """(environments, metacommunity-wide frequency, per-patch frequencies) at the end of each climate stage."""
    species, patch_ids = run['species'], run['patch_ids']
    order = _patch_order(patch_ids)
    envs, meta_freq, patch_freq = [], [], []
    for env, step in stage_samples(run['steps'], env_curve):
        snapshot = species.iloc[int(np.argmin(np.abs(run['steps'] - step)))]
        occupied = snapshot.notna().sum()
        envs.append(env)
        meta_freq.append((snapshot == species_id).sum() / occupied if occupied else np.nan)
        patch_freq.append(_snapshot_patch_frequency(snapshot, patch_ids, order, species_id))
    return np.array(envs), np.array(meta_freq), np.array(patch_freq)


def tipping_point(envs, freqs, level=0.5):
    """The environment at which one branch hands the landscape over: its first crossing of `level`, taken in
    the order the climate stages occur (so a warming branch is read left to right and a cooling branch right
    to left) and interpolated between the two bracketing stages.

    Interpolating is what keeps the two branches comparable. Any rule that reports one end of the crossing
    interval -- the last stage still held, or the first stage after -- offsets warming from cooling by a
    whole stage even where the loop is closed, because the two travel that interval in opposite directions.

    Returns (index of the first stage past the crossing, environment at the crossing), or (None, None) for a
    branch that never crosses `level`."""
    valid = [i for i in range(len(freqs)) if not np.isnan(freqs[i])]
    if len(valid) < 2: return None, None
    started_above = freqs[valid[0]] >= level
    for previous, i in zip(valid, valid[1:]):
        if (freqs[i] >= level) != started_above:
            f0, f1, e0, e1 = freqs[previous], freqs[i], envs[previous], envs[i]
            return i, e0 if f1 == f0 else e0 + (level - f0) * (e1 - e0) / (f1 - f0)
    return None, None


# --------------------------------------------------------------------------- #
# The two ways of flattening the loop (section 3.6)
# --------------------------------------------------------------------------- #
def _linearity(environments, shares):
    """How close a composition curve is to a straight line rather than to a step, as
    1 - (residual about a linear fit) / (residual about a step at the midpoint). ~1 means a diagonal,
    <=0 means the step still describes it better."""
    environments, shares = np.asarray(environments, float), np.asarray(shares, float)
    keep = ~np.isnan(shares)
    if keep.sum() < 3: return np.nan
    environments, shares = environments[keep], shares[keep]
    cold = environments < environments.mean()
    line = np.polyval(np.polyfit(environments, shares, 1), environments)
    step = np.where(cold, shares[cold].mean(), shares[~cold].mean())
    line_error, step_error = np.sum((shares - line) ** 2), np.sum((shares - step) ** 2)
    return 1 - line_error / step_error if step_error > 0 else np.nan


def plot_flattening(runs, warming_env, cooling_env, file_name='example2_flattening.jpg'):
    """One panel per configuration: sp1's share of the metacommunity against the mean environment, walked
    in both directions, with the individual patches as faint dots and a perfect diagonal for reference.
    `runs` is {label: {'warming': run, 'cooling': run}}. The two branches meeting means history stopped
    mattering; them meeting *along the diagonal* means the threshold did too."""
    labels = list(runs)
    fig, axes = plt.subplots(1, len(labels), figsize=(4.4 * len(labels), 4.6), sharey=True, squeeze=False)
    for ax, label in zip(axes[0], labels):
        for direction, env_curve, colour, marker in (('warming', warming_env, 'tab:red', 'o-'), ('cooling', cooling_env, 'tab:blue', 's--')):
            envs, meta_freq, patch_freq = stage_frequencies(runs[label][direction], env_curve)
            for i, env in enumerate(envs):
                ax.plot(np.full(patch_freq.shape[1], env), patch_freq[i], '.', color=colour, alpha=0.15, ms=4)
            ax.plot(envs, meta_freq, marker, color=colour, lw=2, ms=5, label=direction)
        ax.plot([0.2, 0.8], [1, 0], ':', color='gray', lw=1.5, label='a perfect diagonal')
        ax.set(title=label, xlabel='mean environment of the landscape', ylim=(-0.05, 1.05))
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("sp1's share of the occupants")
    axes[0][0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Two ways of losing the loop — and why the dots matter more than the lines', fontsize=12)
    plt.tight_layout()
    plt.savefig(file_name, dpi=110)
    plt.close('all')
    display(Image(file_name))


def flattening_table(runs, warming_env, cooling_env, at_env=0.5):
    """The three numbers that separate the two ways of flattening the loop: how far apart the branches
    still are, how straight the curve is, and -- the one that tells them apart -- whether the individual
    patches sit in the middle or are piled at 0 and 1."""
    rows = []
    for label, directions in runs.items():
        envs, warm_freq, warm_patch = stage_frequencies(directions['warming'], warming_env)
        _, cool_freq, cool_patch = stage_frequencies(directions['cooling'], cooling_env)
        i = int(np.argmin(np.abs(envs - at_env)))
        patches = np.concatenate([warm_patch[i], cool_patch[i]])
        patches = patches[~np.isnan(patches)]
        rows.append([label, '%.2f' % abs(warm_freq[i] - cool_freq[i]), '%.2f' % _linearity(envs, warm_freq), '%.2f' % _linearity(envs, cool_freq),
                     '%.0f%%' % (100 * np.mean((patches > 0.2) & (patches < 0.8))), '%.0f%%' % (100 * np.mean((patches <= 0.2) | (patches >= 0.8)))])
    return attr_table(rows, columns=['configuration', 'gap between the branches at %.1f' % at_env, 'straightness (warming)', 'straightness (cooling)', 'patches half-and-half', 'patches at 0 or 1'])


# --------------------------------------------------------------------------- #
# Replay the recorded landscape as a video
# --------------------------------------------------------------------------- #
def _tile_patches(values, patch_ids, patch_grid, hab_length, hab_width):
    """Fold a (time x microsite) table back into (time, y, x) landscape images.

    `values` is a DataFrame or a plain array whose columns are the microsites in the recorded order;
    `patch_ids` is the patch of every column. Patch k sits at x = k // n_y, y = k % n_y with y growing
    upwards, which is the layout build_metacommunity uses."""
    n_x, n_y = patch_grid
    values = values.to_numpy() if hasattr(values, 'to_numpy') else np.asarray(values)
    order = sorted(set(patch_ids), key=lambda p: int(p.replace('patch', '')))
    canvas = np.full((values.shape[0], n_y * hab_length, n_x * hab_width), np.nan)
    for k, patch in enumerate(order):
        block = values[:, np.asarray(patch_ids) == patch].reshape(-1, hab_length, hab_width)
        x, y = k // n_y, k % n_y
        canvas[:, (n_y - 1 - y) * hab_length:(n_y - y) * hab_length, x * hab_width:(x + 1) * hab_width] = block
    return canvas


def render_environment_species_phenotype_video(run, env_curve, patch_grid, hab_length, hab_width, file_name, title='', fps=4, every=2):
    """One run as a three-panel animation: the environment it is living in, who is living there, and what
    trait they carry. Everything is rebuilt from the two recorded tables, so no simulation is re-run.

    The environment is the one place that needs reconstructing: only its t=0 field is recorded, and
    meta_offset_environmental_values redraws each microsite at every climate step. What is drawn is the
    recorded initial field plus the accumulated offset -- the right spatial pattern and the right mean,
    without the per-microsite re-draw (sd 0.025, invisible at this scale).

    `every` subsamples the recorded snapshots; the file is roughly linear in the number of frames. `fps`
    only sets the playback delay, not the size -- 4 gives a quarter-second per climate snapshot, slow enough
    to read the three panels against each other."""
    species, phenotype, patch_ids = run['species'].iloc[::every], run['phenotype'].iloc[::every], run['patch_ids']
    steps, env_curve = run['steps'][::every], np.asarray(env_curve)[::every]
    environment = np.add.outer(env_curve - env_curve[0], run['env0'].to_numpy())    # initial field, ramped

    canvases = {'environment': _tile_patches(environment, patch_ids, patch_grid, hab_length, hab_width),
                'species': _tile_patches(species, patch_ids, patch_grid, hab_length, hab_width),
                'phenotype': _tile_patches(phenotype, patch_ids, patch_grid, hab_length, hab_width)}

    species_cmap, trait_cmap = plt.get_cmap('coolwarm').copy(), plt.get_cmap('viridis').copy()
    for cmap in (species_cmap, trait_cmap): cmap.set_bad('white')
    panels = [('environment', 'environment', plt.get_cmap('coolwarm'), 0.0, 1.0, 'cold  →  warm'),
              ('species', 'species', species_cmap, 1, 2, 'blue = sp1 (cold)   red = sp2 (warm)   white = empty'),
              ('phenotype', 'phenotype', trait_cmap, 0.0, 1.0, 'the trait each individual carries')]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), constrained_layout=True)
    images = {}
    for ax, (key, name, cmap, vmin, vmax, note) in zip(axes, panels):
        images[key] = ax.imshow(canvases[key][0], cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel(note, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if key != 'species': fig.colorbar(images[key], ax=ax, fraction=0.046, pad=0.02)
    heading = fig.suptitle('', fontsize=13)

    def update(t):
        for key in images: images[key].set_data(canvases[key][t])
        occupied = np.count_nonzero(~np.isnan(canvases['species'][t]))
        heading.set_text('%s\ntime step %d      environment %.2f      %d microsites occupied' % (title, steps[t], env_curve[t], occupied))
        return list(images.values()) + [heading]

    FuncAnimation(fig, update, frames=len(steps), blit=False).save(file_name, writer=PillowWriter(fps=fps))
    plt.close('all')
    return file_name, len(steps)
