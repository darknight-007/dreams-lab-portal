#!/usr/bin/env bash
# Build /mnt/22tb-hdd/datasets/ tree on the dreamslab server.
# Mirrors the per-project category layout of tesseract:/mnt/22tb-hdd/datasets/
# but points each symlink at the actual data location on *this* dreamslab server.
#
# Idempotent: re-running only re-creates symlinks, never deletes data.
#
# Usage: sudo bash build_datasets_tree.sh

set -euo pipefail

ROOT=/mnt/22tb-hdd/datasets
LOG=$ROOT/MANIFEST.log

mkdir -p "$ROOT"
: > "$LOG"

ln_safe() {
  local target="$1" link="$2"
  if [ ! -e "$target" ] && [ ! -L "$target" ]; then
    echo "MISSING target=$target link=$link" | tee -a "$LOG"
    return 0
  fi
  mkdir -p "$(dirname "$link")"
  if [ -L "$link" ] || [ -e "$link" ]; then
    rm -f "$link"
  fi
  ln -s "$target" "$link"
  echo "LN $link -> $target" >> "$LOG"
}

# ---------- AQUATIC ----------
ln_safe /mnt/dreamslab-store/deepgis/deepgis_fish                         "$ROOT/aquatic/deepgis_fish/portal"
ln_safe /mnt/dreamslab-store/hanand/fish_data                             "$ROOT/aquatic/deepgis_fish/hanand_fish_data"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/data_augmentor/datasets/fish    "$ROOT/aquatic/deepgis_fish/sarah_data_augmentor_fish"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/data_augmentor/datasets/guppy   "$ROOT/aquatic/deepgis_fish/sarah_data_augmentor_guppy"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_fish_old                     "$ROOT/aquatic/deepgis_fish_old/portal"

# Bryant's GDCS coral dataset did not survive on this dreamslab server.
# Keep an orphan stub with a README pointing back to tesseract.
mkdir -p "$ROOT/aquatic/gdcs_coral_bryant"
cat > "$ROOT/aquatic/gdcs_coral_bryant/ORPHAN_ON_TESSERACT.md" <<'EOF'
Bryant's GDCS coral dataset (Mask-RCNN-Coral-Species-Training-and-Detection)
is only on tesseract: /mnt/12tb-hdd-B/Bryant_GDCS_datasets/...
No copy is present on this dreamslab server.
EOF

ln_safe /mnt/dreamslab-store/deepgis/deepgis_coral/deepgis_coral          "$ROOT/aquatic/gdcs_coral_harish/portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_aqualink                     "$ROOT/aquatic/gdcs_coral_harish/aqualink"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_sensors                      "$ROOT/aquatic/gdcs_coral_harish/sensors"
ln_safe /mnt/dreamslab-store/hanand/coral-images-png                      "$ROOT/aquatic/gdcs_coral_harish/hanand_coral_images_png"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/coral-models                    "$ROOT/aquatic/gdcs_coral_harish/sarah_coral_models"

# alex_hawaii only exists on tesseract backup (sarah/alex_hawaii). Orphan here.
mkdir -p "$ROOT/aquatic/alex_hawaii_orphan"
cat > "$ROOT/aquatic/alex_hawaii_orphan/ORPHAN_ON_TESSERACT.md" <<'EOF'
alex_hawaii only exists on tesseract:
/mnt/12tb-hdd-B/dreamslab-hdd-bkup/sarah/alex_hawaii
No copy found on this dreamslab server.
EOF

# ---------- SPACE ----------
ln_safe /mnt/dreamslab-store/hanand/download.openuas.us/trained_param_crater  "$ROOT/space/crater_sweep_100epochs/hanand_openuas_trained_param_crater"
# zhiang's crater datasets are tesseract-only — stub
mkdir -p "$ROOT/space/crater_sweep_100epochs"
cat > "$ROOT/space/crater_sweep_100epochs/ORPHAN_ON_TESSERACT.md" <<'EOF'
zhiang_crater / zhiang_Crater / zhiang_iros / zhiang_moon_tiles[_png[_x2]]
live only on tesseract:
  /mnt/12tb-hdd-B/zhiang/zhiang_deep_learning/datasets/{crater,Crater,iros}
  /mnt/12tb-hdd-A/zhiang_annotation_game/mask_rcnn/moon_tiles{,_png,_png_x2}
Only hanand's openuas `trained_param_crater` checkpoint set is mirrored here
(see sibling `hanand_openuas_trained_param_crater` symlink).
EOF

ln_safe /mnt/dreamslab-store/deepgis/deepgis_brent                        "$ROOT/space/deepgis_brent/portal"
ln_safe /mnt/dreamslab-store/hanand/download.openuas.us/brent_dataset.zip "$ROOT/space/deepgis_brent/hanand_openuas_brent_dataset_zip"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_lroc_nacr                    "$ROOT/space/deepgis_lroc_nacr/portal"
# curate every LROC NAC zip under one subdir
LROC_SRC=/mnt/dreamslab-store/hanand/download.openuas.us
for f in lroc-fix-dataset.zip LROCImages.zip lroc_nac_all_jan13.zip \
         lroc_nac_corrected_jan13.zip lroc_nac_dataset_Dec21.zip \
         lroc_nac_Dec_07_v01.zip lroc_nac_Dec_22_v01.zip \
         lroc_nac_Dec_23_v01.zip lroc_nac_Dec_30_v01.zip \
         lroc_nac_Dec_31_v01.zip lroc-nac.zip \
         offset_corrected_lroc_nac.zip M1348265832LE.tif; do
  ln_safe "$LROC_SRC/$f" "$ROOT/space/deepgis_lroc_nacr/hanand_openuas_zips/$f"
done

ln_safe /mnt/dreamslab-store/deepgis/deepgis_mars                         "$ROOT/space/deepgis_mars/portal"
ln_safe /mnt/dreamslab-store/hanand/mars_dataset                          "$ROOT/space/deepgis_mars/hanand_mars_dataset"
for f in mars_all_jan9.zip mars_corrected_jan9.tar.xz mars_Dec_07_v01.zip \
         mars_Dec_24_v01.zip mars_Dec29_v01.zip mars.zip; do
  ln_safe "$LROC_SRC/$f" "$ROOT/space/deepgis_mars/hanand_openuas_zips/$f"
done

ln_safe /mnt/dreamslab-store/deepgis/deepgis_moon                         "$ROOT/space/hanand_stragglers/deepgis_moon_portal"
ln_safe /mnt/dreamslab-store/hanand/MOON_BACKUP                           "$ROOT/space/hanand_stragglers/hanand_MOON_BACKUP"
ln_safe /mnt/dreamslab-store/hanand/deepgis_moon_db_1.tar                 "$ROOT/space/hanand_stragglers/hanand_deepgis_moon_db_1_tar"
ln_safe /mnt/dreamslab-store/hanand/moon_3_july2020.tar                   "$ROOT/space/hanand_stragglers/hanand_moon_3_july2020_tar"
ln_safe /mnt/dreamslab-store/hanand/download.openuas.us/moon-images       "$ROOT/space/hanand_stragglers/hanand_openuas_moon_images"

# ---------- TERRESTRIAL ----------
ln_safe /mnt/22tb-hdd/Samsung_T5_Dios/bishop2019                          "$ROOT/terrestrial/bishop_jezero_field/samsung_t5_bishop2019"
ln_safe /mnt/22tb-hdd/Samsung_T5_Dios/jezero                              "$ROOT/terrestrial/bishop_jezero_field/samsung_t5_jezero"
# zhiang's Eureka/Bishop deep-learning datasets are tesseract-only
mkdir -p "$ROOT/terrestrial/bishop_jezero_field"
cat > "$ROOT/terrestrial/bishop_jezero_field/ORPHAN_ON_TESSERACT.md" <<'EOF'
zhiang_eureka / zhiang_eureka_infer / zhiang_bishop (deep-learning dataset
form) live only on tesseract:
  /mnt/12tb-hdd-B/zhiang/zhiang_deep_learning/datasets/{Eureka,Eureka_infer}
  /mnt/12tb-hdd-B/dreamslab-hdd-bkup/sarah/zhiang_bishop
On this dreamslab server we only have the raw Bishop 2019 UAV field data
(Samsung T5 Dios drive) and the Jezero reference material, linked here as
`samsung_t5_bishop2019` and `samsung_t5_jezero`.
EOF

ln_safe /mnt/dreamslab-store/deepgis/deepgis_agu_hypolith                 "$ROOT/terrestrial/deepgis_agu_hypolith/portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_hypolith                     "$ROOT/terrestrial/deepgis_agu_hypolith/legacy_hypolith"
ln_safe /mnt/tesseract-store/life/hypolith                                "$ROOT/terrestrial/deepgis_agu_hypolith/tesseract_store_life_hypolith"
ln_safe /mnt/tesseract-store/datasets/hypolith_sample_set_throop          "$ROOT/terrestrial/deepgis_agu_hypolith/tesseract_store_hypolith_sample_set_throop"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_agu_litter                   "$ROOT/terrestrial/deepgis_agu_litter/portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_litter                       "$ROOT/terrestrial/deepgis_agu_litter/legacy_litter"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/jdas/agu-litter-line                  "$ROOT/terrestrial/deepgis_agu_litter/jdas_agu_litter_line"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_agu_litter_raw               "$ROOT/terrestrial/deepgis_agu_litter_raw/portal"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_flow                         "$ROOT/terrestrial/deepgis_flow/portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_flow2                        "$ROOT/terrestrial/deepgis_flow/portal_flow2"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_flow3                        "$ROOT/terrestrial/deepgis_flow/portal_flow3"
ln_safe /mnt/dreamslab-store/deepgis/Devin_wildfire_files                 "$ROOT/terrestrial/deepgis_flow/devin_wildfire_files"
ln_safe /mnt/dreamslab-store/deepgis_flow                                 "$ROOT/terrestrial/deepgis_flow/app_deepgis_flow"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_flow_old                     "$ROOT/terrestrial/deepgis_flow_old/portal"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_litter_dynamics              "$ROOT/terrestrial/deepgis_litter_dynamics/portal"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_new_life                     "$ROOT/terrestrial/deepgis_new_life/portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_life                         "$ROOT/terrestrial/deepgis_new_life/legacy_life"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_life_bkup                    "$ROOT/terrestrial/deepgis_new_life/legacy_life_bkup"
ln_safe /mnt/tesseract-store/life/images                                  "$ROOT/terrestrial/deepgis_new_life/tesseract_store_life_images"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_roadkill                     "$ROOT/terrestrial/deepgis_roadkill/portal"

ln_safe /mnt/dreamslab-store/hanand/Gobabeb-1FD68D16-38B5-4B1D-B3F8-AA62A74B4C16_1_201_a  "$ROOT/terrestrial/gobabeb_namib/hanand_gobabeb_tiles_1FD68D16"
ln_safe /mnt/dreamslab-store/hanand/Gobabeb-2B32EB75-534C-47BB-BCAB-313F3DC4B4FC_1_201_a  "$ROOT/terrestrial/gobabeb_namib/hanand_gobabeb_tiles_2B32EB75"
ln_safe /mnt/dreamslab-store/hanand/Gobabeb-2F47464C-4FDA-4E31-8855-2264CFF08CDE_1_201_a  "$ROOT/terrestrial/gobabeb_namib/hanand_gobabeb_tiles_2F47464C"
ln_safe /mnt/22tb-hdd/Samsung_T5_Dios/Gobabeb                             "$ROOT/terrestrial/gobabeb_namib/samsung_t5_gobabeb"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/jdas/agu-hypoliths/Gobabeb            "$ROOT/terrestrial/gobabeb_namib/jdas_agu_hypoliths_gobabeb"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/Documents/Zhiang/rock_traits    "$ROOT/terrestrial/gobabeb_namib/zhiang_rock_traits"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/Documents/Zhiang/C3.tif         "$ROOT/terrestrial/gobabeb_namib/zhiang_C3_tif"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/Documents/Zhiang/C3_mask_v3.tif "$ROOT/terrestrial/gobabeb_namib/zhiang_C3_mask_v3_tif"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/Documents/Zhiang/C3_dem.tif     "$ROOT/terrestrial/gobabeb_namib/zhiang_C3_dem_tif"
ln_safe /mnt/dreamslab-store/deepgis/C3.tif                               "$ROOT/terrestrial/gobabeb_namib/deepgis_C3_tif"
ln_safe /mnt/dreamslab-store/deepgis/C3_mask_v3.tif                       "$ROOT/terrestrial/gobabeb_namib/deepgis_C3_mask_v3_tif"

ln_safe /mnt/dreamslab-store/deepgis/deepgis_rocks                        "$ROOT/terrestrial/legacy-matterport-2018/deepgis_rocks_portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_rocks2                       "$ROOT/terrestrial/legacy-matterport-2018/deepgis_rocks2_portal"
ln_safe /mnt/dreamslab-store/deepgis/deepgis_rocks_static_serving_nginx   "$ROOT/terrestrial/legacy-matterport-2018/deepgis_rocks_static_serving_nginx"
ln_safe /mnt/dreamslab-store/deepgis/rocks_raster_tile_server             "$ROOT/terrestrial/legacy-matterport-2018/rocks_raster_tile_server"
ln_safe /mnt/dreamslab-store/deepgis/rock_tiles.mbtiles                   "$ROOT/terrestrial/legacy-matterport-2018/rock_tiles_mbtiles"
ln_safe /mnt/dreamslab-store/hanand/deepgis_rocks_persistent-storage      "$ROOT/terrestrial/legacy-matterport-2018/hanand_deepgis_rocks_persistent_storage"
ln_safe /mnt/dreamslab-store/hanand/deepgis_rocks_db                      "$ROOT/terrestrial/legacy-matterport-2018/hanand_deepgis_rocks_db"
ln_safe /mnt/dreamslab-store/hanand/deepgis_rocks_db_1.tar                "$ROOT/terrestrial/legacy-matterport-2018/hanand_deepgis_rocks_db_1_tar"
ln_safe /mnt/dreamslab-store/hanand/deepgis-rocks-mediawiki.tar           "$ROOT/terrestrial/legacy-matterport-2018/hanand_deepgis_rocks_mediawiki_tar"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/jdas/data_hdd/datasets/camtraps       "$ROOT/terrestrial/legacy-matterport-2018/jdas_camtraps_dataset"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/jdas/data_hdd/datasets/balloon_dataset "$ROOT/terrestrial/legacy-matterport-2018/jdas_balloon_dataset"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/wildlife                        "$ROOT/terrestrial/legacy-matterport-2018/sarah_wildlife"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/wildlife2                       "$ROOT/terrestrial/legacy-matterport-2018/sarah_wildlife2"
ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/deepgis/static-root/labels      "$ROOT/terrestrial/legacy-matterport-2018/sarah_deepgis_labels"

ln_safe /mnt/22tb-hdd/2tbssdcx-bkup/sarah/data_augmentor/datasets/tornado "$ROOT/terrestrial/tornado/sarah_data_augmentor_tornado"
# zhiang tornado2018/2019 are tesseract-only
cat > "$ROOT/terrestrial/tornado/ORPHAN_ON_TESSERACT.md" <<'EOF'
tornado2018 / tornado2019 live only on tesseract:
/mnt/12tb-hdd-B/dreamslab-hdd-bkup/zhiang/datasets/tornado{2018,2019}
Only sarah's data_augmentor tornado annotation set exists here.
EOF

# cross-link maskrcnn _refs so data and models share annotations
ln_safe /mnt/22tb-hdd/maskrcnn/_refs                                      "$ROOT/_refs"

echo "=== LN summary ==="
grep -c '^LN' "$LOG" | xargs echo "symlinks created ="
grep -c '^MISSING' "$LOG" | xargs echo "missing targets ="
