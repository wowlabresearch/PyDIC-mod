#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==================================================================================
# This file is part of pydic, a free digital correlation suite for computing strain fields.
# 
# Original Author: Damien ANDRE, SPCTS/ENSIL-ENSCI, Limoges France
# <damien.andre@unilim.fr>
# Copyright (C) 2017 Damien ANDRE
#
# [MODIFIED VERSION]
# This version has been customized for the "Experiments in Mechanical Engineering" course @ Hongik University, Seoul, South Korea by Kyung Yun Choi.
# Significant updates include:
# 1. Interactive Hole Masking: Excludes noisy data around the specimen hole.
# 2. Stress Concentration Analysis: Focuses on local strain distribution around a hole.
# ==================================================================================

# ====== INTRODUCTION
# This script demonstrates how to compute material properties (Young's modulus and 
# Poisson's ratio) using Digital Image Correlation (DIC). 
# 
# [Experiment: Open-Hole Tensile Test]
# 1. Specimen: A rectangular plate with a central hole.
# 2. Key Focus: Observing 'Stress Concentration' around the hole.
# 3. Data: Images (.jpg / .bmp / .png) and loading values (.txt) must be stored in the 'img' directory.
#
# ====== HOW TO USE
# 1. Run the script: 'py main.py' (or 'python main.py').
# 2. Interactive Hole Masking: A pop-up window will appear showing the strain field.
# 3. Post-Processing: Once you close the pop-up windows, the script will calculate
#    the mechanical properties and display them in the terminal.


# ====== IMPORTING MODULES
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import os
import cv2
import glob
# import pydic
import sys
sys.path.append('../')
import pydic




#  ====== RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS (STRUCTURED GRID)
correl_wind_size = (80,80) # the size in pixel of the correlation windows
correl_grid_size = (20,20) # the size in pixel of the interval (dx,dy) of the correlation grid
scale_disp = 10
scale_grid = 25 #생성된 grid 이미지 격자간 scale (pixel 단위)
post_mask_interpolation = 'linear'


# read image series and write a separated result file 
pydic.init('./img/*.jpg', correl_wind_size, correl_grid_size, "result.dic")


# and read the result file for computing strain and displacement field from the result file 
pydic.read_dic_file('result.dic', interpolation='spline', strain_type='log', save_image=False, scale_disp=scale_disp, scale_grid=scale_grid, meta_info_file='img/meta-data.txt')


#  ====== OR RUN PYDIC TO COMPUTE DISPLACEMENT AND STRAIN FIELDS (WITH UNSTRUCTURED GRID OPTION)
# note that you can't use the 'spline' or the 'raw' interpolation with unstructured grids 
# please uncomment the next lines if you want to use the unstructured grid options instead of the aligned grid
# pydic.init('./img/*.bmp', correl_wind_size, correl_grid_size, "result.dic", unstructured_grid=(20,5))
# pydic.read_dic_file('result.dic', interpolation='cubic', save_image=True, scale_disp=10, scale_grid=25, meta_info_file='img/meta-data.txt')



#  ====== RESULTS
# Now you can go in the 'img/pydic' directory to see the results :
# - the 'disp', 'grid' and 'marker' directories contain image files
# - the 'result' directory contain raw text csv file where displacement and strain fields are written  



# ======= STANDARD POST-TREATMENT : STRAIN FIELD MAP PLOTTING
# the pydic.grid_list is a list that contains all the correlated grids (one per image)
# the grid objects are the main objects of pydic  
# ======= STANDARD POST-TREATMENT : STRAIN FIELD MAP PLOTTING
last_grid = pydic.grid_list[-1]

# ---------------------------------------------------------
# [추가] 1. 구멍 영역 마스킹 (마우스 동적 선택)
# ---------------------------------------------------------
def select_hole_circle(reference_field, background_image=None):
    ny, nx = reference_field.shape

    # DIC 원본 시편 이미지를 원본 해상도로 표시하고, 클릭 좌표를 그대로 사용
    selection_bg = None
    if isinstance(background_image, str):
        selection_bg = cv2.imread(background_image, cv2.IMREAD_GRAYSCALE)
    elif background_image is not None:
        selection_bg = background_image

    # fallback: 마지막 프레임 이미지 로드
    if selection_bg is None:
        image_files = sorted(glob.glob('./img/*.jpg'))
        if image_files:
            selection_bg = cv2.imread(image_files[-1], cv2.IMREAD_GRAYSCALE)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    if selection_bg is not None:
        ax.imshow(selection_bg, cmap='gray', origin='upper', interpolation='nearest')
        ax.set_title('Hole mask selection (high-res image): click center, then edge')
    else:
        # 원본 이미지가 없으면 기존 strain 맵으로 폴백
        im = ax.imshow(reference_field, cmap='viridis', origin='upper')
        ax.set_title('Hole mask selection (fallback): click center, then edge')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel('Use toolbar zoom/pan if needed, then click center and edge')

    # center 1회 + edge 1회 클릭
    pts = plt.ginput(2, timeout=-1)
    plt.close(fig)

    if len(pts) < 2:
        return None

    (cx, cy), (ex, ey) = pts
    radius = np.hypot(ex - cx, ey - cy)
    if radius <= 0:
        return None

    return cx, cy, radius


def build_circular_mask(grid_x, grid_y, cx, cy, radius):
    # 최종 contourf와 동일한 좌표계(grid_x/grid_y, pixel coord)에서 마스크 생성
    dist = np.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
    return dist <= radius


def recompute_grids_with_hole_as_void(grids, cx, cy, radius, interpolation_method='linear', strain_type='cauchy'):
    # 구멍 내부 추적점은 재료가 없는 것으로 간주하여 보간/변형률 계산 입력에서 제거
    min_points_required = 20

    for idx, grid in enumerate(grids):
        ref_pts = np.asarray(grid.reference_point, dtype=np.float64)
        corr_pts = np.asarray(grid.correlated_point, dtype=np.float64)
        disp_raw = np.asarray(grid.disp, dtype=np.float64)

        if ref_pts.ndim != 2 or ref_pts.shape[1] != 2 or disp_raw.ndim != 2 or disp_raw.shape[1] != 2:
            raise RuntimeError(f'frame {idx}: reference/displacement point format이 예상과 다릅니다.')

        inside_hole = ((ref_pts[:, 0] - cx) ** 2 + (ref_pts[:, 1] - cy) ** 2) <= (radius ** 2)
        finite_pts = np.isfinite(ref_pts[:, 0]) & np.isfinite(ref_pts[:, 1])
        finite_disp = np.isfinite(disp_raw[:, 0]) & np.isfinite(disp_raw[:, 1])
        keep = (~inside_hole) & finite_pts & finite_disp

        kept_count = int(np.count_nonzero(keep))
        if kept_count < min_points_required:
            raise RuntimeError(
                f'frame {idx}: 구멍 제외 후 유효 추적점이 부족합니다 ({kept_count}개). '
                'hole 반경을 줄이거나 grid 밀도를 높이세요.'
            )

        # 시각화에서도 hole 내부 추적점이 보이지 않도록 NaN 처리
        ref_pts[inside_hole] = np.nan
        corr_pts[inside_hole] = np.nan
        grid.reference_point = ref_pts
        grid.correlated_point = corr_pts

        # hole 외부 점만으로 displacement 재보간 -> strain 재계산
        grid.interpolate_displacement(ref_pts[keep], disp_raw[keep], method=interpolation_method)

        if strain_type == 'cauchy':
            grid.compute_strain_field()
        elif strain_type == '2nd_order':
            grid.compute_strain_field_DA()
        elif strain_type == 'log':
            grid.compute_strain_field_log()
        else:
            raise RuntimeError("strain_type은 'cauchy', '2nd_order', 'log' 중 하나여야 합니다.")


def apply_hole_mask_to_grids(grids, hole_mask):
    flat_invalid = hole_mask.reshape(-1)

    for grid in grids:
        if grid.strain_xx.shape != hole_mask.shape:
            raise RuntimeError('grid shape가 프레임마다 달라 동일 hole mask를 적용할 수 없습니다.')

        # 후처리 계산에서 구멍 내부를 완전히 제외
        grid.strain_xx[hole_mask] = np.nan
        grid.strain_yy[hole_mask] = np.nan
        grid.strain_xy[hole_mask] = np.nan

        # disp/grid 이미지에서도 구멍 영역의 격자선이 사라지도록 변위도 NaN 처리
        grid.disp_x[hole_mask] = np.nan
        grid.disp_y[hole_mask] = np.nan

        # marker/disp 벡터 시각화에서 구멍 내부 포인트를 제거
        if hasattr(grid, 'reference_point') and hasattr(grid, 'correlated_point'):
            if len(grid.reference_point) == flat_invalid.size and len(grid.correlated_point) == flat_invalid.size:
                grid.reference_point[flat_invalid] = np.nan
                grid.correlated_point[flat_invalid] = np.nan


def export_masked_outputs(grids, scale_disp, scale_grid):
    for grid in grids:
        grid.draw_marker_img()
        grid.draw_disp_img(scale_disp)
        grid.draw_grid_img(scale_grid)
        grid.write_result()


selection = select_hole_circle(last_grid.strain_xx, background_image=last_grid.image)
if selection is not None:
    cx, cy, radius = selection
    recompute_grids_with_hole_as_void(
        pydic.grid_list,
        cx,
        cy,
        radius,
        interpolation_method=post_mask_interpolation,
        strain_type='log'
    )

    hole_mask = build_circular_mask(last_grid.grid_x, last_grid.grid_y, cx, cy, radius)
    print(f"[알림] Hole mask 적용: center=({cx:.2f}, {cy:.2f}), radius={radius:.2f}")
    print(f"[알림] 구멍 제외 후 재보간/재계산 완료 (interpolation={post_mask_interpolation})")
    valid_region_mask = ~hole_mask

    # 선택한 동일 마스크를 모든 프레임에 적용- 후속 계산까지 일관되게 반영
    removed_per_grid = int(np.count_nonzero(hole_mask))
    total_per_grid = hole_mask.size
    print(f"[알림] 프레임당 제외 격자점: {removed_per_grid}/{total_per_grid}")

    apply_hole_mask_to_grids(pydic.grid_list, hole_mask)
    export_masked_outputs(pydic.grid_list, scale_disp=scale_disp, scale_grid=scale_grid)
    print('[알림] 마스킹이 반영된 disp/grid/marker/result 파일을 다시 저장했습니다.')
else:
    hole_mask = None
    valid_region_mask = np.ones_like(last_grid.strain_xx, dtype=bool)
    print("[경고] Hole mask 선택이 취소되어 마스킹 없이 계산을 진행합니다.")

# 마스킹이 적용된 맵 그리기
last_grid.plot_field(last_grid.strain_xx, 'xx strain (Hole Masked)')
last_grid.plot_field(last_grid.strain_yy, 'yy strain (Hole Masked)')
plt.show()

# ======== NON-STANDARD POST-TREATMENT : COMPUTE ELASTIC CONSTANTS (E & Nu)
try:
    meta_data = np.genfromtxt('img/meta-data.txt', skip_header=1)
    force = meta_data[:, 1]
    force = force[:len(pydic.grid_list)] 
    print(f"[알림] 메타데이터에서 {len(force)}개의 힘(Force) 데이터를 성공적으로 읽었습니다.")
except Exception as e:
    print(f"[경고] meta-data.txt 파일을 읽는 중 에러가 발생했습니다: {e}")
    force = np.zeros(len(pydic.grid_list))

sample_width     = 0.012
sample_thickness = 0.002
stress = force/(sample_width * sample_thickness)

# ---------------------------------------------------------
# [추가] 2. 응력 집중(Concentrated Stress) 분석
# 하중 방향: yy (strain_yy가 양수)
# 횡방향:   xx (strain_xx는 Poisson 수축으로 음수)
# ---------------------------------------------------------

peak_strain_yy_list = []      # 구멍 주변 최대 변형률 (95th percentile, outlier 제거)
ave_strain_xx_list = []       # 횡방향 평균 변형률 (Nu 계산용)
nominal_strain_yy_list = []   # far-field 공칭 변형률 
for grid in pydic.grid_list:
    local_yy = grid.strain_yy[valid_region_mask]
    local_xx = grid.strain_xx[valid_region_mask]

    local_yy = local_yy[np.isfinite(local_yy)]
    local_xx = local_xx[np.isfinite(local_xx)]

    if local_yy.size == 0 or local_xx.size == 0:
        peak_strain_yy_list.append(np.nan)
        ave_strain_xx_list.append(np.nan)
        nominal_strain_yy_list.append(np.nan)
    else:
        # 95th percentile로 경계 outlier 제거
        peak_strain_yy_list.append(np.percentile(local_yy, 95))
        ave_strain_xx_list.append(np.mean(local_xx))

        # far-field nominal: 상단/하단 1/4 행 → yy 인장이 균일한 영역
        ny, nx = grid.strain_yy.shape
        far_field_mask = np.zeros((ny, nx), dtype=bool)
        far_field_mask[:ny // 4, :] = True       # 상단 1/4
        far_field_mask[3 * ny // 4:, :] = True   # 하단 1/4
        if hole_mask is not None:
            far_field_mask &= ~hole_mask
        nom_vals = grid.strain_yy[far_field_mask]
        nom_vals = nom_vals[np.isfinite(nom_vals)]
        nominal_strain_yy_list.append(np.mean(nom_vals) if nom_vals.size > 0 else np.nan)

peak_strain_yy_list = np.array(peak_strain_yy_list)
ave_strain_xx_list = np.array(ave_strain_xx_list)
nominal_strain_yy_list = np.array(nominal_strain_yy_list)

# ============================================================
# [실습] 응력 집중 계수 (Kt) 계산
# 아래 변수들을 활용하여 Kt를 직접 계산하여 보고서에 기술:
#   peak_strain_yy_list   : 각 프레임의 최대 국부 변형률 
#   nominal_strain_yy_list: 각 프레임의 공칭 변형률 
#   Kt = ??
# ============================================================

# Direct Kt calculation (last frame)
#Kt = ??

# Linear regression to calculate final Young's modulus (loading direction yy)
valid = np.isfinite(peak_strain_yy_list) & np.isfinite(ave_strain_xx_list) & np.isfinite(stress)
if np.count_nonzero(valid) < 2:
    raise RuntimeError('유효한 데이터 포인트가 부족하여 회귀를 수행할 수 없습니다.')

E, intercept, r_value, p_value, std_err = stats.linregress(peak_strain_yy_list[valid], stress[valid])
Nu, intercept, r_value, p_value, std_err = stats.linregress(peak_strain_yy_list[valid], -ave_strain_xx_list[valid])

print ("\n응력 집중을 고려한 결과 (Concentrated Stress Area) :")
print ("  => 영률 E = {:.2f} GPa".format(E*1e-9))
print ("  => 푸아송비 Nu = {:.2f}".format(Nu))
print ("  => 최종 단계 peak strain_yy (95th pct) = {:.5f}".format(peak_strain_yy_list[-1]))
print ("  => 최종 단계 nominal strain_yy (far-field) = {:.5f}".format(nominal_strain_yy_list[-1]))
#print ("  => 응력 집중 계수 Kt = {:.3f}".format(Kt))
