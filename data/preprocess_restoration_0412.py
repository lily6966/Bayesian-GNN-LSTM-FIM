"""
Preprocess FIM_with_restoration_75000_0412.csv for GNN-RNN training.

NA handling strategy
--------------------
Continuous variables
  <5% NA  : median imputation within (Bay, Month), fallback to overall median
  5-30% NA : median imputation within (Bay, Zone, Month), fallback chain to
             (Bay, Month) → overall median; PLUS a binary "was_missing" indicator
  >30% NA  : binary indicator only (e.g. DominantVeg → has_dominant_veg flag)

Categorical variables
  Encode to integers; NAs become a dedicated "Unknown" (or last) category
  so the model can learn to treat missingness as its own signal.

Features (environmental + community species)
--------------------------------------------
  Environmental  (3): Temperature, Salinity, DissolvedO2
  Restoration    (8): n_projects, Acres_Restored, targeted_habitat_number,
                      Restoration_technique_number, primary_habitat,
                      secondary_habitat, Technique_first, Technique_second
  Habitat cont.  (9): BottomVegCover, bMud, bSan, bStr, bUnk,
                      SAV, Alg, Non, HA, TH, RU
  Shoreline      (4): Nat, Ove, Str, Eme  (+4 was_na indicators at 12% NA)
  Distance       (2): Dist_to_Shore, Dist_to_MHTM  (+2 was_na indicators at 25-27% NA)
  Water/effort   (4): Secchi_depth, TotalShoreCover, CloudCover, StartDepth, Effort
                      (+1 was_na for TotalShoreCover at 17.4%)
  Secchi binary  (1): Secchi_on_bottom_bin
  Habitat cat.   (4): bveg_enc, DominantBot_enc, DominantShore_enc, Tide_enc
  Extra binary   (1): has_dominant_veg  (proxy for seagrass presence)
  Bycatch       (14): Byc_Bivalve, Byc_Seagrass, Byc_Algae, Byc_Debris, Byc_SAV,
                      Byc_Crabs, Byc_Substrate, Byc_Jelly, Byc_Unkn, Byc_NonAquatVeg,
                      Byc_Invert, Byc_BiotStruct, Byc_MiscBio, Byc_MiscABio
  Community sp. (~92): all species without _A/_R/_SA suffix (log1p-transformed sums)
                       ALSO included as prediction targets

Targets
-------
  All species: community species (no suffix) + life-stage species (_A/_R, after _SA→_A merge)

Aggregation (haul → station × month × year)
---------------------------------------------
  Species counts : sum → log1p
  Continuous     : mean (NaN-aware)
  Restoration    : max (cumulative)
  Categorical    : mode (most frequent)
  NA indicators  : max (if any haul was NA → group was NA)

Outputs
-------
  data/FIM_restoration_0412_stations_monthly.npz
  data/FIM_restoration_0412_station_metadata.pkl
  data/FIM_restoration_0412_species_names.pkl
"""

import os, pickle
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

# ── Common name lookup (scientific name → common name) ────────────────────────
COMMON_NAMES = {
    'Archosargus probatocephalus': 'Sheepshead',
    'Callinectes sapidus':         'Blue Crab',
    'Centropomus undecimalis':     'Common Snook',
    'Cynoscion nebulosus':         'Spotted Seatrout',
    'Lagodon rhomboides':          'Pinfish',
    'Lutjanus griseus':            'Gray Snapper',
    'Mycteroperca microlepis':     'Gag Grouper',
    'Sciaenops ocellatus':         'Red Drum',
}

LIFE_STAGE_LABELS = {
    '_A':  'Adult',
    '_a':  'Adult',
    '_R':  'Recruit',
    '_r':  'Recruit',
    '_J':  'Juvenile',
    '_j':  'Juvenile',
    '_SA': 'Sub-adult',
}

def col_to_common(col):
    """Return 'Common Name (Stage)' for a species column, or the raw col name."""
    for sfx, stage in LIFE_STAGE_LABELS.items():
        if col.endswith(sfx):
            sci = col[: -len(sfx)]
            common = COMMON_NAMES.get(sci, sci)
            return f"{common} ({stage})"
    return col

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(DATA_DIR, 'FIM_with_restoration_75000_0412.csv')
GRIDS_PATH = os.path.join(DATA_DIR, 'FIM_UniverseGrids.RData')

# Minimum hauls for a (Bay, Zone, Grid) to be retained as a station
MIN_HAULS_PER_STATION = 50

# ── Target species: columns ending _A, _R, _J, _SA ───────────────────────────
def get_species_cols(df):
    return [c for c in df.columns
            if c.endswith('_A') or c.endswith('_R') or c.endswith('_J')
            or c.endswith('_SA') or c.endswith('_a') or c.endswith('_r')
            or c.endswith('_j')]

# ── Categorical encoding maps ──────────────────────────────────────────────────
BVEG_MAP = {'Non': 0, 'SAV': 1, 'Alg': 2, 'SAVAlg': 3}     # Unknown → 4
DOMINANTBOT_CATEGORIES = ['Mud', 'MudSan', 'MudSanStr', 'MudStr',
                           'San', 'SanStr', 'Str', 'Unknown']
DOMINANTSHORE_CATEGORIES = ['Eme', 'EmeOve', 'Ove', 'Str', 'StrOve',
                             'Unknown']
TIDE_MAP = {'LF': 0, 'MF': 1, 'HF': 2, 'LR': 3, 'MR': 4, 'HR': 5}  # Unknown→6

# ── Non-target (community) species: those NOT ending in _A, _R, _SA ──────────
# These are aggregated sum → log1p and used as input features
NON_META_COLS = {
    'Reference', 'n_projects', 'Acres_Restored', 'targeted_habitat_number',
    'Restoration_technique_number', 'primary_habitat', 'secondary_habitat',
    'Technique_first', 'Technique_second', 'Bay', 'Type', 'Year', 'Month',
    'Gear', 'Zone', 'Grid', 'Effort', 'Longitude', 'Latitude', 'StartTime',
    'StartDepth', 'BycatchQuantity', 'BottomVegCover', 'TotalShoreCover',
    'Secchi_depth', 'Secchi_on_bottom', 'Dist_to_Shore', 'Dist_to_ShoreType',
    'Dist_to_MHTM', 'CloudCover', 'Tide', 'bMud', 'bSan', 'bStr', 'bUnk',
    'DominantBot', 'SAV', 'Alg', 'Non', 'HA', 'TH', 'RU', 'bveg', 'DominantVeg',
    'Nat', 'Ove', 'Str', 'Eme', 'DominantShore', 'Temperature', 'Salinity',
    'DissolvedO2',
}

# ── Continuous feature definitions ────────────────────────────────────────────
# (name, agg_method, na_threshold_for_indicator)
# agg_method: 'mean' | 'max'
CONT_FEATURES = [
    # Environmental
    ('Temperature',               'mean', 0.05),
    ('Salinity',                  'mean', 0.05),
    ('DissolvedO2',               'mean', 0.05),
    # Restoration (cumulative → max)
    ('n_projects',                'max',  0.0),
    ('Acres_Restored',            'max',  0.0),
    ('targeted_habitat_number',   'max',  0.0),
    ('Restoration_technique_number', 'max', 0.0),
    ('primary_habitat',           'max',  0.0),
    ('secondary_habitat',         'max',  0.0),
    ('Technique_first',           'max',  0.0),
    ('Technique_second',          'max',  0.0),
    # Habitat continuous
    ('BottomVegCover',            'mean', 0.05),
    ('bMud',                      'mean', 0.05),
    ('bSan',                      'mean', 0.05),
    ('bStr',                      'mean', 0.0),   # 0% NA
    ('bUnk',                      'mean', 0.0),   # 0% NA
    ('SAV',                       'mean', 0.05),
    ('Alg',                       'mean', 0.05),
    ('Non',                       'mean', 0.05),
    ('HA',                        'mean', 0.05),
    ('TH',                        'mean', 0.05),
    ('RU',                        'mean', 0.05),
    # Shoreline type fractions (12% NA → median imputation + indicator)
    ('Nat',                       'mean', 0.05),
    ('Ove',                       'mean', 0.05),
    ('Str',                       'mean', 0.05),
    ('Eme',                       'mean', 0.05),
    # Distance features (25-27% NA → median imputation + indicator)
    ('Dist_to_Shore',             'mean', 0.05),
    ('Dist_to_MHTM',              'mean', 0.05),
    # Water / effort
    ('Secchi_depth',              'mean', 0.05),
    ('CloudCover',                'mean', 0.0),   # 0% NA
    ('StartDepth',                'mean', 0.0),   # 0% NA
    ('Effort',                    'mean', 0.0),   # 0% NA
    # Bycatch (0.2% NA → simple imputation, no indicator)
    ('Byc_Bivalve',               'mean', 0.05),
    ('Byc_Seagrass',              'mean', 0.05),
    ('Byc_Algae',                 'mean', 0.05),
    ('Byc_Debris',                'mean', 0.05),
    ('Byc_SAV',                   'mean', 0.05),
    ('Byc_Crabs',                 'mean', 0.05),
    ('Byc_Substrate',             'mean', 0.05),
    ('Byc_Jelly',                 'mean', 0.05),
    ('Byc_Unkn',                  'mean', 0.05),
    ('Byc_NonAquatVeg',           'mean', 0.05),
    ('Byc_Invert',                'mean', 0.05),
    ('Byc_BiotStruct',            'mean', 0.05),
    ('Byc_MiscBio',               'mean', 0.05),
    ('Byc_MiscABio',              'mean', 0.05),
]

CAT_FEATURES = ['bveg_enc', 'DominantBot_enc', 'DominantShore_enc', 'Tide_enc']
EXTRA_BINARY = ['has_dominant_veg', 'Secchi_on_bottom_bin']


# ── Imputation helpers ────────────────────────────────────────────────────────
def impute_median(series, df, group_cols):
    """Fill NAs using median within group_cols, then overall median."""
    out = series.copy()
    if out.isnull().sum() == 0:
        return out
    grp_med = df.groupby(group_cols)[series.name].transform('median')
    out = out.fillna(grp_med)
    overall = series.median()
    out = out.fillna(overall)
    return out


def mode_agg(x):
    """Mode aggregation; returns NaN if all values are NaN."""
    vals = x.dropna()
    if len(vals) == 0:
        return np.nan
    m = scipy_mode(vals, keepdims=True)
    return m.mode[0]


# ── Main preprocessing ─────────────────────────────────────────────────────────
def preprocess(csv_path=CSV_PATH, zero_rd_gear020=False, out_suffix=''):
    print("Loading data …")
    df = pd.read_csv(csv_path)
    print(f"  Raw shape: {df.shape}")

    # ── Optional: Gear-020 Red Drum filter (Run G data prep) ──────────────────
    # Red Drum (Sciaenops ocellatus) is poorly sampled by Gear 020 (seine).
    # Zero out Red Drum counts in Gear-020 hauls so the model doesn't learn
    # noise from those observations. All other species remain intact.
    if zero_rd_gear020 and 'Gear' in df.columns:
        rd_cols = [c for c in df.columns if c.startswith('Sciaenops ocellatus')]
        if rd_cols:
            mask020 = df['Gear'] == 20
            n_zeroed_rows = int(mask020.sum())
            df.loc[mask020, rd_cols] = 0
            print(f"  [filter] Zeroed Red Drum {rd_cols} in {n_zeroed_rows:,} "
                  f"Gear-020 hauls (other species unchanged)")

    # ── 0. Identify community species (no life-stage suffix) ──────────────────
    _tgt_sfx_early = ('_A', '_R', '_J', '_a', '_r', '_j', '_SA')
    all_target_cols = set(get_species_cols(df))
    community_sp_cols = sorted([
        c for c in df.columns
        if c not in NON_META_COLS
        and c not in all_target_cols
        and not c.startswith('Byc_')
        and not any(c.endswith(sfx) for sfx in _tgt_sfx_early)
        and not c.endswith('_enc') and not c.endswith('_bin')
        and pd.api.types.is_numeric_dtype(df[c])
    ])
    print(f"  Community species (no life-stage suffix): {len(community_sp_cols)}")

    # ── 1. Encode categorical columns ─────────────────────────────────────────
    print("Encoding categorical variables …")

    df['bveg_enc'] = df['bveg'].map(BVEG_MAP).fillna(4).astype(int)

    df['DominantBot_filled'] = df['DominantBot'].fillna('Unknown')
    bot_cat = pd.Categorical(df['DominantBot_filled'],
                             categories=DOMINANTBOT_CATEGORIES)
    df['DominantBot_enc'] = bot_cat.codes.astype(int)

    df['DominantShore_filled'] = df['DominantShore'].fillna('Unknown')
    shore_cat = pd.Categorical(df['DominantShore_filled'],
                               categories=DOMINANTSHORE_CATEGORIES)
    df['DominantShore_enc'] = shore_cat.codes.astype(int)

    df['Tide_enc'] = df['Tide'].map(TIDE_MAP).fillna(6).astype(int)

    # DominantVeg: >30% NA → binary indicator only
    df['has_dominant_veg'] = (~df['DominantVeg'].isnull()).astype(float)

    # Secchi_on_bottom: YES=1, NO=0
    df['Secchi_on_bottom_bin'] = df['Secchi_on_bottom'].map({'YES': 1.0, 'NO': 0.0}).fillna(0.0)

    # ── 2. Load FIM Universe Grids → fixed spatial centroids ─────────────────
    print("Loading FIM Universe Grids …")
    import pyreadr
    universe = pyreadr.read_r(GRIDS_PATH)['UnivGrids']
    # Fixed centroid per (Bay, Zone, Grid) — average in case of rare duplicates
    grid_centroids = (universe.groupby(['Bay', 'Zone', 'Grid'])
                      .agg(grid_lat=('Latitude', 'mean'),
                           grid_lon=('Longitude', 'mean'))
                      .reset_index())
    print(f"  {len(grid_centroids)} universe grid cells loaded")

    # Join fixed centroids onto haul data
    df['Grid'] = pd.to_numeric(df['Grid'], errors='coerce').astype('Int64')
    df = df.merge(grid_centroids, on=['Bay', 'Zone', 'Grid'], how='left')
    n_missing = df['grid_lat'].isnull().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} hauls had no matching universe grid → dropped")
        df = df[df['grid_lat'].notna()].copy()

    # ── 3. Define stations: Bay × Zone × Grid (≥ MIN_HAULS) ──────────────────
    print(f"Defining stations (Bay × Zone × Grid, min {MIN_HAULS_PER_STATION} hauls) …")
    haul_counts = df.groupby(['Bay', 'Zone', 'Grid']).size()
    eligible_grids = haul_counts[haul_counts >= MIN_HAULS_PER_STATION].reset_index()[
        ['Bay', 'Zone', 'Grid']]
    df = df.merge(eligible_grids, on=['Bay', 'Zone', 'Grid'], how='inner')
    print(f"  Hauls after grid filtering: {len(df)}")

    stations = (df[['Bay', 'Zone', 'Grid']].drop_duplicates()
                .sort_values(['Bay', 'Zone', 'Grid']).reset_index(drop=True))
    stations['station_id'] = stations.index
    df = df.merge(stations, on=['Bay', 'Zone', 'Grid'], how='left')
    n_stations = len(stations)
    print(f"  {n_stations} stations (Bay×Zone×Grid grids)")

    # Use fixed universe grid centroids for metadata (not haul-average)
    station_meta = stations.merge(grid_centroids, on=['Bay', 'Zone', 'Grid'], how='left')
    station_meta = station_meta.rename(columns={'grid_lat': 'lat', 'grid_lon': 'lon'})

    # Impute continuous columns (group within Bay×Zone×Month)
    print("Imputing continuous variables …")
    na_indicator_cols = []

    for col, agg, na_thresh in CONT_FEATURES:
        if col not in df.columns:
            print(f"  WARNING: {col} not found in CSV, skipping")
            continue
        na_pct = df[col].isnull().mean()
        if na_pct == 0:
            continue
        print(f"  {col}: {na_pct*100:.1f}% NA", end='')
        if na_pct > na_thresh and na_thresh > 0:
            ind_col = f'{col}_was_na'
            df[ind_col] = df[col].isnull().astype(float)
            na_indicator_cols.append(ind_col)
            print(f" → indicator '{ind_col}' added", end='')
        df[col] = impute_median(df[col], df, ['Bay', 'Zone', 'Month'])
        print()

    # ── 4. Target species columns (_SA kept as separate targets) ─────────────────
    sp_cols = sorted(get_species_cols(df))  # includes _A, _R, _J, _SA, _a, _r, _j
    print(f"\n  {len(sp_cols)} target species (_SA kept separate):")
    for col in sp_cols:
        print(f"    {col:<45} → {col_to_common(col)}")

    # Community species: numeric columns without _A/_R/_SA suffix, not metadata/bycatch
    # → used as encoded input features only (not targets)
    # Ordered by gear selectivity (Gear 20/23 = seine; Gear 160 = drop/cast net):
    #   prey_community    (47): caught EXCLUSIVELY by seine (Gear 20 and/or 23, never 160)
    #   nonprey_community (45): ever caught by Gear 160 (may also appear in seine)
    # This grouping is used as two separate community tokens in the model.
    _target_suffixes = ('_A', '_R', '_J', '_a', '_r', '_j', '_SA')
    _exclude_computed = {'station_id', 'node_id', 'grid_lat', 'grid_lon',
                         'DominantBot_filled', 'DominantShore_filled'}
    _all_community = sorted([
        c for c in df.columns
        if c not in NON_META_COLS
        and c not in _exclude_computed
        and not any(c.endswith(sfx) for sfx in _target_suffixes)
        and not c.startswith('Byc_')
        and not c.endswith('_enc') and not c.endswith('_bin')
        and not c.endswith('_was_na')
        and c != 'has_dominant_veg'
        and pd.api.types.is_numeric_dtype(df[c])
    ])

    # Gear-based split derived from FIM raw data (Gear column):
    #   seine_only   = species whose non-zero rows use ONLY Gear 20 or 23 (never 160)
    #   non_seine    = species that appear at least once in Gear 160 hauls
    _seine_only = [
        'Achirus lineatus', 'Alburnops petersoni', 'Anchoa hepsetus',
        'Anchoa mitchilli', 'Bathygobius soporator', 'Chasmodes saburrae',
        'Ctenogobius boleosoma', 'Cynoscion arenarius', 'Cyprinella venusta',
        'Cyprinodon variegatus', 'Enneacanthus gloriosus', 'Eucinostomus spp.',
        'Farfantepenaeus duorarum', 'Farfantepenaeus spp.', 'Floridichthys carpio',
        'Fundulus chrysotus', 'Fundulus grandis', 'Fundulus seminolis',
        'Fundulus xenicus', 'Gambusia holbrooki', 'Gobiosoma bosc',
        'Gobiosoma robustum', 'Gobiosoma spp.', 'Heterandria formosa',
        'Hippocampus zosterae', 'Labidesthes vanhyningi', 'Lepomis macrochirus',
        'Lepomis microlophus', 'Lepomis punctatus', 'Lepomis spp.',
        'Litopenaeus setiferus', 'Lucania goodei', 'Lucania parva',
        'Lutjanus synagris', 'Membras martinica', 'Menidia spp.',
        'Microgobius gulosus', 'Micropterus salmoides', 'Notemigonus crysoleucas',
        'Poecilia latipinna', 'Prionotus scitulus', 'Prionotus tribulus',
        'Stephanolepis hispida', 'Strongylura spp.', 'Symphurus plagiusa',
        'Syngnathus scovelli', 'Trinectes maculatus',
    ]   # 47 species

    _seine_only_set = set(_seine_only)
    _non_seine      = [c for c in _all_community if c not in _seine_only_set]  # 45 species

    # Final ordering: prey_community first (47), then nonprey_community (45)
    _prey_in_data    = [c for c in _seine_only if c in set(_all_community)]
    _nonprey_in_data = [c for c in _non_seine  if c in set(_all_community)]
    community_sp_cols_updated = _prey_in_data + _nonprey_in_data

    print(f"  {len(community_sp_cols_updated)} community species → encoded as input features")
    print(f"    prey_community (seine-only): {len(_prey_in_data)}")
    print(f"    nonprey_community (Gear 160): {len(_nonprey_in_data)}")

    # ── 5. Build aggregation spec & named feature groups ─────────────────────
    cont_col_names = [c for c, _, _ in CONT_FEATURES if c in df.columns]

    # Named feature groups (order matters — defines token layout for Transformer)
    # Environmental: water-column measurements + water clarity + tidal state
    env_cols         = [c for c in ['Temperature','Salinity','DissolvedO2',
                                    'Secchi_depth','Secchi_on_bottom_bin','Tide_enc'] if c in df.columns]
    restoration_cols = [c for c in ['n_projects','Acres_Restored','targeted_habitat_number',
                                    'Restoration_technique_number','primary_habitat',
                                    'secondary_habitat','Technique_first','Technique_second']
                        if c in df.columns]
    # Habitat: substrate, vegetation fractions + dominant vegetation categoricals
    habitat_cols     = [c for c in ['BottomVegCover','bMud','bSan','bStr','bUnk',
                                    'SAV','Alg','Non','HA','TH','RU',
                                    'bveg_enc','DominantBot_enc','has_dominant_veg']
                        if c in df.columns]
    # Shoreline + distance: cover fractions, dominant shore type, and proximity
    shoreline_cols   = [c for c in ['Nat','Ove','Str','Eme','DominantShore_enc',
                                    'Dist_to_Shore','Dist_to_MHTM'] if c in df.columns]
    distance_cols    = []  # merged into shoreline_cols
    water_cols       = [c for c in ['CloudCover','StartDepth','Effort'] if c in df.columns]
    bycatch_cols     = [c for c in cont_col_names if c.startswith('Byc_')]
    # NA indicators: missingness flags only (separate group, no categorical mixing)
    na_ind_cols      = na_indicator_cols

    # Ordered feature column list
    all_feat_cols = (env_cols + restoration_cols + habitat_cols + shoreline_cols +
                     water_cols + bycatch_cols + na_ind_cols)

    # Build group boundary dict: name → (start, end) indices within feat_cols_final
    # (community species and Ybar groups added later in model)
    def _boundaries(cols_list, offset=0):
        boundaries = []
        pos = offset
        for cols in cols_list:
            boundaries.append(pos)
            pos += len(cols)
        boundaries.append(pos)
        return boundaries

    group_col_lists = [env_cols, restoration_cols, habitat_cols, shoreline_cols,
                       water_cols, bycatch_cols, na_ind_cols]
    group_names     = ['environmental', 'restoration', 'habitat', 'shoreline',
                       'water_effort', 'bycatch', 'na_indicators']
    # Community group boundaries (appended after env block in main.py group_slices)
    n_env_total       = len(all_feat_cols)   # 58
    n_prey_community  = len(_prey_in_data)   # 47
    n_nonprey_community = len(_nonprey_in_data)  # 45
    bounds = _boundaries(group_col_lists)
    feature_groups = [(name, bounds[i], bounds[i+1])
                      for i, name in enumerate(group_names)]
    print(f"\n  Feature groups:")
    for name, s, e in feature_groups:
        print(f"    {name:<20} cols {s:>3}–{e-1:>3}  ({e-s} features)")

    agg_spec = {}
    col_agg_map = {c: agg for c, agg, _ in CONT_FEATURES}
    for col in cont_col_names:
        agg_spec[col] = 'mean' if col_agg_map.get(col, 'mean') == 'mean' else 'max'
    # Override restoration cols to max
    for col, agg_method, _ in CONT_FEATURES:
        if col in df.columns and agg_method == 'max':
            agg_spec[col] = 'max'
    for col in CAT_FEATURES:
        agg_spec[col] = mode_agg
    for col in EXTRA_BINARY + na_ind_cols:
        agg_spec[col] = 'max'
    for col in sp_cols:
        agg_spec[col] = 'sum'
    for col in community_sp_cols_updated:
        agg_spec[col] = 'sum'

    # ── 6. Aggregate hauls → (station_id, Year, Month) ────────────────────────
    print("\nAggregating to station × year × month …")
    group_cols = ['station_id', 'Year', 'Month']
    agg_df = df.groupby(group_cols).agg(agg_spec).reset_index()

    # log1p-transform target species
    for col in sp_cols:
        if col in agg_df.columns:
            agg_df[col] = np.log1p(agg_df[col].clip(lower=0))
    # log1p-transform community species (as encoded features)
    for col in community_sp_cols_updated:
        if col in agg_df.columns:
            agg_df[col] = np.log1p(agg_df[col].clip(lower=0))

    print(f"  Aggregated rows: {len(agg_df)}")

    # ── 7. node_id = station_id * 12 + (Month - 1) ───────────────────────────
    agg_df['node_id'] = agg_df['station_id'] * 12 + (agg_df['Month'] - 1)

    # ── 8. Assemble NPZ matrix ────────────────────────────────────────────────
    # Column order: [node_id, year, sp_0..sp_N, env_feats..., community_sp_feats...]
    feat_cols_final = all_feat_cols + community_sp_cols_updated
    output_cols = ['node_id', 'Year'] + sp_cols + feat_cols_final

    # Fill remaining NAs in environmental features
    for col in all_feat_cols:
        if col in agg_df.columns and agg_df[col].isnull().any():
            med = agg_df[col].median()
            agg_df[col] = agg_df[col].fillna(med if not np.isnan(med) else 0.0)
    # Community species features: fill NaN with 0
    for col in community_sp_cols_updated:
        if agg_df[col].isnull().any():
            agg_df[col] = agg_df[col].fillna(0.0)

    data_matrix = agg_df[output_cols].values.astype(np.float32)
    feature_start = 2 + len(sp_cols)

    print(f"\n  Data matrix shape: {data_matrix.shape}")
    print(f"  feature_start: {feature_start}  ({len(feat_cols_final)} features)")
    print(f"    Env+habitat+bycatch features: {len(all_feat_cols)}")
    print(f"    Community species (encoded): {len(community_sp_cols_updated)}")
    print(f"  Target species ({len(sp_cols)}): {sp_cols}")

    # ── 9. Save outputs ────────────────────────────────────────────────────────
    suffix = out_suffix or ''
    npz_path  = os.path.join(DATA_DIR, f'FIM_restoration_0412{suffix}_stations_monthly.npz')
    meta_path = os.path.join(DATA_DIR, f'FIM_restoration_0412{suffix}_station_metadata.pkl')
    spp_path  = os.path.join(DATA_DIR, f'FIM_restoration_0412{suffix}_species_names.pkl')

    np.savez(npz_path, data=data_matrix, feature_start=np.array([feature_start]))

    meta_rows = []
    for _, row in station_meta.iterrows():
        for m in range(12):
            meta_rows.append({
                'station_id': int(row['station_id']),
                'node_id':    int(row['station_id']) * 12 + m,
                'Bay':        row['Bay'],
                'Zone':       row['Zone'],
                'Grid':       int(row['Grid']),
                'lat':        row['lat'],
                'lon':        row['lon'],
                'month':      m + 1,
            })
    meta_df = pd.DataFrame(meta_rows)

    with open(meta_path, 'wb') as f:
        pickle.dump(meta_df, f)
    with open(spp_path, 'wb') as f:
        pickle.dump({
            'target_species':      sp_cols,                    # _A/_R/_SA/_J targets
            'feature_species':     community_sp_cols_updated,  # community species (prey first, nonprey second)
            'n_env_features':      len(all_feat_cols),          # total env/habitat/bycatch/cat features
            'feature_groups':      feature_groups,              # [(name, start, end), ...] within env block
            'group_names':         group_names,                 # ordered list of group names
            'n_prey_community':    n_prey_community,            # 47 seine-only species (first in feature_species)
            'n_nonprey_community': n_nonprey_community,         # 45 Gear-160 species  (second)
        }, f)

    print(f"\n✓  NPZ  → {npz_path}")
    print(f"✓  Meta → {meta_path}")
    print(f"✓  Spp  → {spp_path}")

    # ── 10. Sanity checks ─────────────────────────────────────────────────────
    print("\n── Sanity checks ──")
    print(f"  Years in data: {sorted(agg_df['Year'].unique())}")
    print(f"  Node IDs range: {int(agg_df['node_id'].min())}–{int(agg_df['node_id'].max())}")
    print(f"  Feature NaN remaining: {np.isnan(data_matrix[:, feature_start:]).sum()}")
    print(f"  Species NaN remaining: {np.isnan(data_matrix[:, 2:feature_start]).sum()}")
    max_y = data_matrix[:, 2:feature_start].max()
    print(f"  Max species log1p value: {max_y:.4f}")

    # Feature summary
    print(f"\n  Feature breakdown:")
    print(f"    env+restoration+habitat+bycatch: {len(all_feat_cols)}")
    print(f"    community species (encoded): {len(community_sp_cols_updated)}")
    print(f"    TOTAL features (in_dim): {len(feat_cols_final)}")

    return npz_path, meta_path, spp_path, feat_cols_final


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--zero_rd_gear020', action='store_true', default=False,
                   help='Zero Red Drum (Sciaenops ocellatus) counts in Gear-020 hauls.')
    p.add_argument('--out_suffix', type=str, default='',
                   help='Suffix appended to output NPZ/PKL filenames (e.g. "G").')
    args = p.parse_args()
    preprocess(zero_rd_gear020=args.zero_rd_gear020, out_suffix=args.out_suffix)
