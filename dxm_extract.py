#!/usr/bin/env python3
"""
adjust_dxm.py
Generate an adjusted winter-wheat (dxm) vector so that within each village the dxm area
matches the target area field (种植面) within tolerance.
Outputs: workspace\adjusted_dxm.gpkg and a per-village report printed to stdout and saved as CSV.

Notes:
- Requires: rasterio, geopandas, shapely, numpy, sklearn (optional), rasterio.features
- Strategy: compute NDVI if possible; if existing dxm vector present, train RandomForest using
  raster bands + NDVI as features and rasterized dxm as labels. Otherwise use NDVI threshold.
"""
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import rasterio
    from rasterio.mask import mask
    from rasterio.features import shapes
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from scipy.ndimage import gaussian_filter, label as ndi_label, binary_opening, binary_closing, find_objects
    import gc
except Exception as e:
    print("Missing required packages:", e)
    print("Please install rasterio, geopandas, shapely, numpy, scikit-learn, pandas, scipy and retry.")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / 'workspace'
WORKSPACE.mkdir(exist_ok=True)

# Helper: find first file with extensions
def find_first(dirpath, exts):
    p = Path(dirpath)
    if not p.exists():
        return None
    for e in exts:
        for f in p.glob(f'*{e}'):
            return f
    return None

raster_path = find_first(ROOT / 'image', ['.tif', '.tiff', '.img'])
village_path = find_first(ROOT / 'border', ['.shp', '.geojson', '.gpkg'])
dxm_path = find_first(ROOT / 'dxm', ['.shp', '.geojson', '.gpkg'])

print('Raster:', raster_path)
print('Village vector:', village_path)
print('Existing dxm vector (for training):', dxm_path)

if raster_path is None or village_path is None:
    print('Required data not found. Place image in image/ and village vector in border/.')
    sys.exit(1)

# Read village polygons
villages = gpd.read_file(str(village_path))
# Attempt to find target area field (Chinese name '种植面') or common alternatives
target_field_candidates = ['种植面', 'target', 'plant_area', 'area', '种植面积']
field = None
for c in target_field_candidates:
    if c in villages.columns:
        field = c
        break
if field is None:
    # fallback to 'plant_area' not present: use area of village as proxy and warn
    print('Warning: no target area field found (looked for', target_field_candidates, ').')
    print('Script will use village polygon area as target (not ideal).')

# Open raster
with rasterio.open(str(raster_path)) as src:
    raster_crs = src.crs
    transform = src.transform
    pixel_area = abs(transform.a * transform.e)
    meta = src.meta.copy()
    bands_count = src.count
    print('Raster crs:', raster_crs, 'bands:', bands_count)

# Reproject villages to raster crs
if villages.crs != raster_crs:
    villages = villages.to_crs(raster_crs)

# If dxm vector exists, read it and compute current dxm area per village by precise clipping
current_dxm_areas = {}
dxm_gdf = None
if dxm_path is not None:
    try:
        dxm_gdf = gpd.read_file(str(dxm_path))
        if dxm_gdf.crs != raster_crs:
            dxm_gdf = dxm_gdf.to_crs(raster_crs)
        # ensure valid polygons
        dxm_gdf['geometry'] = dxm_gdf.geometry.buffer(0)
        for idx, vrow in villages.iterrows():
            try:
                vill_gdf = gpd.GeoDataFrame([vrow], crs=villages.crs, geometry=[vrow.geometry])
                intersect = gpd.overlay(dxm_gdf, vill_gdf, how='intersection')
                if not intersect.empty:
                    # areas in projected CRS are in square meters
                    area = intersect.geometry.area.sum()
                else:
                    area = 0.0
            except Exception:
                area = 0.0
            current_dxm_areas[idx] = area
        print('Computed current dxm area per village from dxm vector.')
    except Exception as e:
        print('Warning: failed to read/prepare dxm vector for area stats:', e)
        dxm_gdf = None

# Read raster into array (all bands)
with rasterio.open(str(raster_path)) as src:
    arr = src.read()  # shape (bands, h, w)
    profile = src.profile
    # convert to float32 to keep memory usage predictable
    arr = arr.astype('float32', copy=False)

# Compute NDVI if bands >=4 (assume band3=red, band4=nir)
ndvi = None
if bands_count >= 4:
    red = arr[2]
    nir = arr[3]
    denom = (nir + red)
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    print('NDVI calculated using bands 3 (red) and 4 (nir).')
else:
    print('NDVI not available (less than 4 bands).')

# Do not build full feature cube (memory heavy). Use arr (bands,h,w) and ndvi separately.
h, w = arr.shape[1], arr.shape[2]

# Simple cloud detection: bright visible pixels with low NDVI
cloud_mask = np.zeros((h, w), dtype=bool)
try:
    if arr.shape[0] >= 3:
        # arr[:3] shape (3,h,w) -> mean across bands axis=0
        vis_mean = np.nanmean(arr[:3], axis=0)
        # use high-percentile threshold to detect very bright pixels (likely clouds)
        valid_vis = vis_mean[~np.isnan(vis_mean)]
        if valid_vis.size > 0:
            bright_thresh = float(np.percentile(valid_vis, 98))
        else:
            bright_thresh = np.inf
        if ndvi is not None:
            cloud_mask = (vis_mean >= bright_thresh) & (ndvi < 0.2)
        else:
            cloud_mask = (vis_mean >= bright_thresh)
        print('Detected cloud pixels:', int(cloud_mask.sum()))
    else:
        print('Insufficient bands to compute simple cloud mask.')
except Exception as e:
    print('Cloud mask computation failed:', e)
    cloud_mask = np.zeros((h, w), dtype=bool)

# If dxm vector exists, rasterize it for labels and train RF
use_rf = False
prob_map = None
if dxm_path is not None:
    try:
        # use dxm_gdf if pre-read, otherwise read from file
        dxm = dxm_gdf if (dxm_gdf is not None) else gpd.read_file(str(dxm_path))
        if dxm is None:
            raise ValueError('dxm_gdf is None')
        if dxm.crs != raster_crs:
            dxm = dxm.to_crs(raster_crs)
        # rasterize dxm: burn value 1 where dxm exists
        from rasterio.features import rasterize
        shapes_iter = ((geom, 1) for geom in dxm.geometry)
        label = rasterize(shapes_iter, out_shape=(h, w), transform=transform, fill=0, dtype='uint8')
        # Sample pixels where label is 0 or 1 and not nodata
        # arr shape (bands, h, w) -> any across bands gives (h, w)
        valid_mask = np.any(~np.isnan(arr), axis=0)
        sample_mask = valid_mask & ((label == 0) | (label == 1))
        # exclude cloud pixels from training
        if cloud_mask is not None:
            sample_mask = sample_mask & (~cloud_mask)
        # Create training dataset from arr (avoid building full feature cube)
        lab_flat = label.ravel()
        y = lab_flat
        mask = (y == 0) | (y == 1)
        # valid_mask: any non-nan across bands
        valid_mask = np.any(~np.isnan(arr), axis=0).ravel()
        mask = mask & valid_mask
        if mask.sum() > 0:
            # build X from band layers
            X_list = [arr[b].ravel()[mask] for b in range(arr.shape[0])]
            if ndvi is not None:
                X_list.append(ndvi.ravel()[mask])
            X = np.vstack(X_list).T
            y = y[mask]
            if len(y) > 100:
                use_rf = True
            else:
                print('Not enough training pixels from existing dxm vector; skipping supervised RF.')
        else:
            print('No valid training pixels found in raster; skipping supervised RF.')
    except Exception as e:
        print('Error preparing supervised data:', e)

if use_rf:
    print('Training RandomForest classifier using existing dxm as labels...')
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # downsample for speed
    if len(y) > 20000:
        X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=20000, stratify=y, random_state=42)
    else:
        X_train, y_train = X, y
    clf.fit(X_train, y_train)
    # predict probability for class 1 across image in chunks to limit memory
    prob_map = np.full((h, w), -1.0, dtype='float32')
    # choose chunk size ~500k pixels
    chunk_pixels = 500000
    chunk_h = max(1, int(chunk_pixels / float(w)))
    try:
        for r0 in range(0, h, chunk_h):
            r1 = min(h, r0 + chunk_h)
            # build feature matrix for this block: shape (bands, r_h, w)
            block = arr[:, r0:r1, :]
            bh = r1 - r0
            flat_block = block.reshape(block.shape[0], -1).T  # (bh*w, bands)
            if ndvi is not None:
                ndv_block = ndvi[r0:r1, :].ravel()[:, None]
                X_block = np.concatenate([flat_block, ndv_block], axis=1)
            else:
                X_block = flat_block
            # mask clouds
            cloud_flat = cloud_mask[r0:r1, :].ravel()
            valid_idx = ~cloud_flat
            if valid_idx.any():
                try:
                    preds = clf.predict_proba(X_block[valid_idx])[:, 1]
                    flat_probs = prob_map[r0:r1, :].ravel()
                    flat_probs[valid_idx] = preds
                    prob_map[r0:r1, :] = flat_probs.reshape(bh, w)
                except Exception:
                    # on any failure, leave -1 values
                    pass
    except Exception as e:
        print('Chunked prediction failed, falling back to whole-image predict:', e)
        try:
            # fallback (may use more memory)
            X_all = []
            for b in range(arr.shape[0]):
                X_all.append(arr[b].ravel())
            X_all = np.vstack(X_all).T
            if ndvi is not None:
                X_all = np.concatenate([X_all, ndvi.ravel()[:, None]], axis=1)
            flat_cloud = cloud_mask.flatten()
            X_all[flat_cloud, :] = 0
            probs = clf.predict_proba(X_all)[:, 1]
            prob_map = probs.reshape(h, w)
        except Exception as e2:
            print('Fallback prediction failed:', e2)
    # mark cloud pixels as excluded
    try:
        prob_map[cloud_mask] = -1
    except Exception:
        pass
    print('Supervised classification completed.')
else:
    # fallback to NDVI thresholding
    if ndvi is None:
        print('No NDVI and no supervised model: cannot classify. Exiting.')
        sys.exit(1)
    thresh = 0.3
    print(f'Using NDVI threshold {thresh} for probability proxy.')
    # scale NDVI to [0,1] as proxy for probability
    prob_map = (ndvi - thresh) / (1 - thresh)
    prob_map = np.clip(prob_map, 0, 1)
    try:
        prob_map[cloud_mask] = -1
    except Exception:
        pass

# Free large intermediate arrays to reduce memory pressure before per-village processing
try:
    del arr
    gc.collect()
except Exception:
    pass

# For each village, create mask by selecting connected patches (object-based) until reaching target area
final_mask = np.zeros((h, w), dtype='uint8')
report_rows = []

# Smooth probability map to enhance spatial continuity (object-based selection benefits)
try:
    # masked gaussian smoothing to preserve edges while reducing noise
    valid_mask = (prob_map >= 0).astype('float32')
    prob_filled = np.where(prob_map >= 0, prob_map, 0.0).astype('float32')
    sigma = 1.0
    smoothed = gaussian_filter(prob_filled, sigma=sigma)
    norm = gaussian_filter(valid_mask, sigma=sigma)
    norm[norm == 0] = 1e-6
    prob_map = (smoothed / norm)
    prob_map = np.clip(prob_map, 0, 1)
except Exception:
    pass

# Create an affine transform helper to compute pixel centers and areas already have pixel_area
for idx, row in villages.iterrows():
    geom = row.geometry
    # mask probability to village
    try:
        # Build a raster mask for the village from its geometry
        from rasterio import features as rfeatures
        mask_arr = rfeatures.rasterize([(mapping(geom), 1)], out_shape=(h, w), transform=transform, fill=0, dtype='uint8')
    except Exception as e:
        print('Error masking raster for village', idx, e)
        continue

    village_probs = prob_map.copy()
    village_probs[mask_arr == 0] = -1  # exclude outside village

    # Determine target area
    if field and field in villages.columns and pd.notnull(row[field]):
        target = float(row[field])
    else:
        # use polygon area
        target = geom.area

    # Build candidate mask using a probability threshold. Use adaptive fallback if no patches found.
    min_patch_pixels = max(5, int(np.floor(100.0 / pixel_area)))  # ignore fragments smaller than ~100 m2
    thresh = 0.1
    if ndvi is not None:
            ndvi_vmask = (ndvi >= 0.08)
            candidate = (village_probs > thresh) & (mask_arr == 1) & ndvi_vmask
    else:
            candidate = (village_probs > thresh) & (mask_arr == 1)
    labeled, num_patches = ndi_label(candidate)

        # If no patches found with initial threshold, broaden candidate to include all village pixels
    if num_patches == 0:
            if ndvi is not None:
                ndvi_vmask = (ndvi >= 0.08)
                candidate = (village_probs >= 0) & (mask_arr == 1) & ndvi_vmask
            else:
                candidate = (village_probs >= 0) & (mask_arr == 1)
            labeled, num_patches = ndi_label(candidate)

    # Gather patches' stats (collect all, keep small ones for high-confidence test)
    patches_all = []
    flat_probs = village_probs.flatten()
    mask_flat = mask_arr.flatten()
    for pid in range(1, num_patches + 1):
        patch_mask = (labeled == pid)
        if not patch_mask.any():
            continue
        patch_flat_idx = np.flatnonzero(patch_mask.flatten())
        patch_count = patch_flat_idx.size
        patch_mean = float(np.mean(flat_probs[patch_flat_idx]))
        patches_all.append({'id': pid, 'mean_prob': patch_mean, 'count': patch_count, 'flat_idx': patch_flat_idx})

    # Unified scoring for all patches with a small bonus for high-confidence cores (no forced inclusion)
    high_conf_threshold = 0.75
    bonus_value = 0.15
    area_tolerance = 1000.0  # m2 allowed soft overrun (reduced to avoid over-aggregation)

    # Filter patches: keep patches >= min_patch_pixels, but also allow small patches if they are high-confidence
    candidate_patches = [p for p in patches_all if (p['count'] >= min_patch_pixels) or (p['mean_prob'] >= high_conf_threshold)]

    # Compute score: mean_prob (with bonus for high confidence) penalized slightly by size
    for p in candidate_patches:
        bonus = bonus_value if p['mean_prob'] >= high_conf_threshold else 0.0
        p['score'] = (p['mean_prob'] + bonus) / np.log1p(p['count'])

    # Sort by unified score (desc)
    candidate_patches.sort(key=lambda x: x['score'], reverse=True)

    # Accumulate patches respecting a soft cap (target + area_tolerance). Partial inclusion allowed for the final patch.
    selected_idx = []
    cum_area = 0.0
    for p in candidate_patches:
        # stop if we've already reached soft cap
        if cum_area >= target + area_tolerance:
            break
        p_area = p['count'] * pixel_area
        remaining_cap = max(0.0, target + area_tolerance - cum_area)
        if p_area <= remaining_cap:
            # include entire patch
            selected_idx.extend(p['flat_idx'].tolist())
            cum_area += p_area
            # if we've reached at least the target (within tolerance), we can stop
            if cum_area >= target - 1e-9:
                break
        else:
            # partial inclusion: include only as many pixels as fit within remaining_cap (prefer high-prob pixels)
            take_pixels = int(np.floor(remaining_cap / pixel_area + 1e-9))
            if take_pixels <= 0:
                # cannot include any pixel from this patch without exceeding cap
                continue
            patch_probs = flat_probs[p['flat_idx']]
            order = np.argsort(patch_probs)[::-1]
            take = p['flat_idx'][order[:take_pixels]]
            selected_idx.extend(take.tolist())
            cum_area += len(take) * pixel_area
            break

    # If still below target (e.g., not enough candidate patches), fall back to pixel-based filling within village but do not exceed target
    if cum_area + 1e-9 < target:
        all_village_flat = np.flatnonzero(mask_flat == 1)
        already = np.array(selected_idx, dtype=int) if len(selected_idx) > 0 else np.array([], dtype=int)
        remaining_flat = np.setdiff1d(all_village_flat, already)
        if remaining_flat.size > 0:
            rem_probs = flat_probs[remaining_flat]
            order = np.argsort(rem_probs)[::-1]
            need_pixels = int(np.ceil((target - cum_area) / pixel_area))
            take = remaining_flat[order[:need_pixels]]
            selected_idx.extend(take.tolist())
            cum_area += len(take) * pixel_area

    # mark in final_mask
    rr = np.array(selected_idx, dtype=int)
    if rr.size > 0:
        ys = rr // w
        xs = rr % w
        final_mask[ys, xs] = 1
    achieved = rr.size * pixel_area
    diff = achieved - target
    report_rows.append({'village_idx': idx, 'target_area_m2': target, 'achieved_area_m2': achieved, 'diff_m2': diff})

# Post-process final_mask to remove thin linear artifacts and tiny objects
try:
    bin_mask = final_mask.astype(bool)
    # small opening then modest closing to remove thin gaps/lines without over-smoothing
    bin_mask = binary_opening(bin_mask, structure=np.ones((2, 2)))
    bin_mask = binary_closing(bin_mask, structure=np.ones((3, 3)))

    # remove very small connected components (noise)
    labeled_pp, num_pp = ndi_label(bin_mask)
    if num_pp > 0:
        objs = find_objects(labeled_pp)
        # minimum object size in pixels (e.g., corresponding to ~100 m2)
        min_obj_m2 = 100.0
        min_obj_pixels = max(1, int(np.floor(min_obj_m2 / pixel_area)))
        remove_labels = []
        for lab_idx, sl in enumerate(objs, start=1):
            if sl is None:
                continue
            comp = (labeled_pp[sl] == lab_idx)
            comp_size = int(comp.sum())
            # compute bounding box dims for elongation check
            bbox_h = sl[0].stop - sl[0].start
            bbox_w = sl[1].stop - sl[1].start
            if comp_size < min_obj_pixels:
                remove_labels.append(lab_idx)
                continue
            # remove very elongated thin objects (likely stripe artifacts)
            if min(bbox_h, bbox_w) > 0:
                elongation = max(bbox_h, bbox_w) / min(bbox_h, bbox_w)
                bbox_area = bbox_h * bbox_w
                solidity = comp_size / bbox_area if bbox_area > 0 else 1.0
                # remove very elongated thin objects or very low-solidity components (likely stripe artifacts)
                if (elongation >= 6 and comp_size < (min_obj_pixels * 10)) or (solidity < 0.15 and comp_size < (min_obj_pixels * 50)):
                    remove_labels.append(lab_idx)
        if len(remove_labels) > 0:
            for rl in remove_labels:
                labeled_pp[labeled_pp == rl] = 0
        bin_mask = labeled_pp > 0
    final_mask = bin_mask.astype('uint8')
except Exception:
    # if any post-processing fails, keep original final_mask
    pass

# Vectorize final_mask
mask_bool = final_mask == 1
shapes_gen = shapes(mask_bool.astype('uint8'), mask=None, transform=transform)
polys = []
vals = []
for geom, val in shapes_gen:
    if val == 1:
        polys.append(shape(geom))
        vals.append(1)

if not polys:
    print('No polygons created. Exiting.')
    sys.exit(1)

out_gdf = gpd.GeoDataFrame({'val': vals, 'geometry': polys}, crs=raster_crs)
# split polygons precisely by village boundaries using spatial intersection
vill = villages.reset_index().rename(columns={'index':'village_idx'})[['village_idx','geometry']]

# perform precise intersection so that pieces belong only to a single village
try:
    split = gpd.overlay(out_gdf, vill, how='intersection')
except Exception as e:
    # fallback: spatial join then clip per village
    print('overlay failed, falling back to per-village clipping:', e)
    pieces = []
    for vidx, vrow in vill.iterrows():
        inter = out_gdf.clip(vrow.geometry)
        if not inter.empty:
            inter['village_idx'] = vrow['village_idx']
            pieces.append(inter)
    if pieces:
        split = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=out_gdf.crs)
    else:
        split = gpd.GeoDataFrame(columns=['val','geometry','village_idx'], crs=out_gdf.crs)

# fix topology and explode multiparts into singlepart polygons
if not split.empty:
    split['geometry'] = split.geometry.buffer(0)
    try:
        split = split.explode(index_parts=False).reset_index(drop=True)
    except TypeError:
        split = split.explode().reset_index(drop=True)
    split = split[split.geometry.notnull() & (split.geometry.area > 0)].copy()
    split = split[split.geometry.type.isin(['Polygon','MultiPolygon'])].copy()

    # drop exact-geometry duplicates using WKB hex
    def _geom_wkb_hex(g):
        if g is None:
            return None
        try:
            attr = getattr(g, 'wkb_hex', None)
            if callable(attr):
                return attr()
            if isinstance(attr, str):
                return attr
            wkb = getattr(g, 'wkb', None)
            if wkb is not None:
                try:
                    return wkb.hex()
                except Exception:
                    return None
            return None
        except Exception:
            return None

    split['wkb'] = split.geometry.apply(_geom_wkb_hex)
    split = split.drop_duplicates(subset='wkb').drop(columns='wkb').reset_index(drop=True)

    # compute area in m2 (assume projected CRS)
    if split.crs is not None and split.crs.is_projected:
        split['area_m2'] = split.geometry.area
    else:
        split = split.to_crs(epsg=3857)
        split['area_m2'] = split.geometry.area
        split = split.to_crs(raster_crs)

    final_gdf = split[['village_idx','area_m2','geometry']].copy()
else:
    final_gdf = gpd.GeoDataFrame(columns=['village_idx','area_m2','geometry'], crs=raster_crs)

# Save to workspace
out_path = WORKSPACE / 'adjusted_dxm.gpkg'
final_gdf.to_file(str(out_path), layer='adjusted_dxm', driver='GPKG')
report_df = pd.DataFrame(report_rows)
report_csv = WORKSPACE / 'adjust_report.csv'
report_df.to_csv(str(report_csv), index=False)

# Print summary and warn villages exceeding 3000 m2
print('\nPer-village report saved to', report_csv)
for r in report_rows:
    if abs(r['diff_m2']) > 3000:
        print('WARNING: village', r['village_idx'], 'diff', r['diff_m2'], 'm2 > 3000')

print('Output vector:', out_path)
print('Done.')
