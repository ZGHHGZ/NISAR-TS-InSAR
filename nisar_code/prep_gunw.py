#!/usr/bin/env python3
"""Compatibility wrapper for MintPy prep_nisar on current ISCE3 GUNW outputs."""

from __future__ import annotations
import h5py
import glob
import os
import re
import sys
import tempfile
from pathlib import Path
import numpy as np
from osgeo import gdal
from mintpy.cli.prep_nisar import cmd_line_parse
import mintpy.prep_nisar as prep_nisar

DATE_PAIR_RE = re.compile(r"(?P<d1>\d{8})_(?P<d2>\d{8})")


def get_date_pairs(filenames: list[str]) -> list[str]:
    date12_list: list[str] = []
    for fname in filenames:
        stem = Path(fname).stem
        match = DATE_PAIR_RE.search(stem)
        if not match:
            raise ValueError(f"Cannot infer date pair from filename: {fname}")
        date12_list.append(f"{match.group('d1')}_{match.group('d2')}")
    return date12_list


def update_dataset_paths(polarization: str = "HH") -> None:
    for key, value in list(prep_nisar.DATASETS.items()):
        if value and "POL" in value:
            prep_nisar.DATASETS[key] = value.replace("POL", polarization)


def compute_common_grid(
    input_files: list[str], bbox: tuple[float, float, float, float] | None = None
) -> tuple[prep_nisar.np.ndarray, prep_nisar.np.ndarray, tuple[float, float, float, float]]:
    west_list: list[float] = []
    east_list: list[float] = []
    south_list: list[float] = []
    north_list: list[float] = []
    x_steps: list[float] = []
    y_steps: list[float] = []

    for file in input_files:
        with prep_nisar.h5py.File(file, "r") as ds:
            xcoord = ds[prep_nisar.DATASETS["xcoord"]][:]
            ycoord = ds[prep_nisar.DATASETS["ycoord"]][:]
            west_list.append(float(prep_nisar.np.min(xcoord)))
            east_list.append(float(prep_nisar.np.max(xcoord)))
            south_list.append(float(prep_nisar.np.min(ycoord)))
            north_list.append(float(prep_nisar.np.max(ycoord)))
            x_steps.append(float(ds[prep_nisar.DATASETS["xSpacing"]][()]))
            y_steps.append(float(ds[prep_nisar.DATASETS["ySpacing"]][()]))

    west = max(west_list)
    east = min(east_list)
    south = max(south_list)
    north = min(north_list)
    if bbox is not None:
        west = max(west, bbox[0])
        south = max(south, bbox[1])
        east = min(east, bbox[2])
        north = min(north, bbox[3])
    if west >= east or south >= north:
        raise ValueError("No common overlap exists among the selected GUNW files")

    x_step = min(abs(step) for step in x_steps)
    y_step = min(abs(step) for step in y_steps)
    width = int(prep_nisar.np.floor((east - west) / x_step)) + 1
    length = int(prep_nisar.np.floor((north - south) / y_step)) + 1
    if width < 2 or length < 2:
        raise ValueError("Common overlap is too small after grid alignment")

    xcoord_common = west + prep_nisar.np.arange(width) * x_step
    ycoord_common = north - prep_nisar.np.arange(length) * y_step
    bounds = (
        float(xcoord_common[0]),
        float(ycoord_common[-1]),
        float(xcoord_common[-1]),
        float(ycoord_common[0]),
    )
    return xcoord_common, ycoord_common, bounds


def resample_to_common_grid(
    data: prep_nisar.np.ndarray,
    src_xcoord: prep_nisar.np.ndarray,
    src_ycoord: prep_nisar.np.ndarray,
    dst_xcoord: prep_nisar.np.ndarray,
    dst_ycoord: prep_nisar.np.ndarray,
    method: str = "nearest",
    fill_value: float = prep_nisar.np.nan,
) -> prep_nisar.np.ndarray:
    dst_y_2d, dst_x_2d = prep_nisar.np.meshgrid(dst_ycoord, dst_xcoord, indexing="ij")
    points = prep_nisar.np.stack((dst_y_2d.flatten(), dst_x_2d.flatten()), axis=-1)
    interpolator = prep_nisar.RegularGridInterpolator(
        (src_ycoord, src_xcoord),
        data,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )
    return interpolator(points).reshape(len(dst_ycoord), len(dst_xcoord))


def extract_common_grid_metadata(
    input_files: list[str],
    bbox: tuple[float, float, float, float] | None = None,
    polarization: str = "HH",
) -> tuple[dict[str, object], prep_nisar.np.ndarray, prep_nisar.np.ndarray]:
    update_dataset_paths(polarization)
    meta_file = input_files[0]
    meta: dict[str, object] = {}

    with prep_nisar.h5py.File(meta_file, "r") as ds:
        pixel_height = float(ds[prep_nisar.DATASETS["ySpacing"]][()])
        pixel_width = float(ds[prep_nisar.DATASETS["xSpacing"]][()])
        meta["EPSG"] = int(ds[prep_nisar.DATASETS["epsg"]][()])
        meta["WAVELENGTH"] = (
            prep_nisar.SPEED_OF_LIGHT / ds[prep_nisar.PROCESSINFO["centerFrequency"]][()]
        )
        meta["ORBIT_DIRECTION"] = ds[prep_nisar.PROCESSINFO["orbit_direction"]][()].decode("utf-8")
        meta["POLARIZATION"] = polarization
        meta["ALOOKS"] = ds[prep_nisar.DATASETS["azimuth_look"]][()]
        meta["RLOOKS"] = ds[prep_nisar.DATASETS["range_look"]][()]
        meta["PLATFORM"] = ds[prep_nisar.PROCESSINFO["platform"]][()].decode("utf-8")
        meta["STARTING_RANGE"] = float(
            prep_nisar.np.nanmin(ds[prep_nisar.PROCESSINFO["rdr_slant_range"]][()])
        )
        start_time = prep_nisar.datetime.datetime.strptime(
            ds[prep_nisar.PROCESSINFO["start_time"]][()].decode("utf-8")[0:26],
            "%Y-%m-%dT%H:%M:%S.%f",
        )
        end_time = prep_nisar.datetime.datetime.strptime(
            ds[prep_nisar.PROCESSINFO["end_time"]][()].decode("utf-8")[0:26],
            "%Y-%m-%dT%H:%M:%S.%f",
        )

    t_mid = start_time + (end_time - start_time) / 2.0
    meta["CENTER_LINE_UTC"] = (
        t_mid - prep_nisar.datetime.datetime(t_mid.year, t_mid.month, t_mid.day)
    ).total_seconds()
    meta["HEIGHT"] = 747000

    xcoord_common, ycoord_common, bounds = compute_common_grid(input_files, bbox)
    meta["X_FIRST"] = float(xcoord_common[0])
    meta["Y_FIRST"] = float(ycoord_common[0])
    meta["X_STEP"] = pixel_width
    meta["Y_STEP"] = pixel_height
    meta["WIDTH"] = len(xcoord_common)
    meta["LENGTH"] = len(ycoord_common)
    meta["bbox"] = ",".join(str(b) for b in bounds)
    if meta["EPSG"] == 4326:
        meta["X_UNIT"] = meta["Y_UNIT"] = "degree"
    else:
        meta["X_UNIT"] = meta["Y_UNIT"] = "meters"
        epsg = str(meta["EPSG"])
        meta["UTM_ZONE"] = epsg[3:] + ("N" if epsg.startswith("326") else "S")
    meta["EARTH_RADIUS"] = prep_nisar.EARTH_RADIUS
    meta["RANGE_PIXEL_SIZE"] = abs(pixel_width)
    meta["AZIMUTH_PIXEL_SIZE"] = abs(pixel_height)
    return meta, xcoord_common, ycoord_common


def prepare_geometry_common_grid(
    outfile: str,
    meta_file: str,
    metadata: dict[str, object],
    dem_file: str,
    mask_file: str | None,
    xcoord_common: prep_nisar.np.ndarray,
    ycoord_common: prep_nisar.np.ndarray,
) -> dict[str, object]:
    print("-" * 50)
    print(f"preparing geometry file: {outfile}")

    dem_subset_array, slant_range, incidence_angle, mask = read_and_interpolate_geometry_on_grid(
        meta_file=meta_file,
        dem_file=dem_file,
        mask_file=mask_file,
        xcoord_common=xcoord_common,
        ycoord_common=ycoord_common,
    )

    meta = dict(metadata)
    meta["FILE_TYPE"] = "geometry"
    meta["STARTING_RANGE"] = float(prep_nisar.np.nanmin(slant_range))
    ds_name_dict = {
        "height": [prep_nisar.np.float32, dem_subset_array.shape, dem_subset_array],
        "incidenceAngle": [prep_nisar.np.float32, incidence_angle.shape, incidence_angle],
        "slantRangeDistance": [prep_nisar.np.float32, slant_range.shape, slant_range],
    }
    if mask_file:
        ds_name_dict["waterMask"] = [prep_nisar.np.bool_, mask.shape, mask.astype(bool)]
    prep_nisar.writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)
    return meta


def prepare_stack_common_grid(
    outfile: str,
    input_files: list[str],
    metadata: dict[str, object],
    date12_list: list[str],
    xcoord_common: prep_nisar.np.ndarray,
    ycoord_common: prep_nisar.np.ndarray,
) -> str:
    print("-" * 50)
    print(f"preparing ifgramStack file: {outfile}")
    num_pair = len(input_files)
    rows = len(ycoord_common)
    cols = len(xcoord_common)
    print(f"number of inputs/unwrapped interferograms: {num_pair}")

    date12_arr = prep_nisar.np.array([x.split("_") for x in date12_list], dtype=prep_nisar.np.bytes_)
    drop_ifgram = prep_nisar.np.ones(num_pair, dtype=prep_nisar.np.bool_)
    ds_name_dict = {
        "date": [date12_arr.dtype, (num_pair, 2), date12_arr],
        "bperp": [prep_nisar.np.float32, (num_pair,), prep_nisar.np.zeros(num_pair, dtype=prep_nisar.np.float32)],
        "dropIfgram": [prep_nisar.np.bool_, (num_pair,), drop_ifgram],
        "unwrapPhase": [prep_nisar.np.float32, (num_pair, rows, cols), None],
        "coherence": [prep_nisar.np.float32, (num_pair, rows, cols), None],
        "connectComponent": [prep_nisar.np.float32, (num_pair, rows, cols), None],
    }
    meta = dict(metadata)
    meta["FILE_TYPE"] = "ifgramStack"
    prep_nisar.writefile.layout_hdf5(outfile, ds_name_dict, metadata=meta)

    print(f"writing data to HDF5 file {outfile} with a mode ...")

    with prep_nisar.h5py.File(outfile, "a") as f:
        prog_bar = prep_nisar.ptime.progressBar(maxValue=num_pair)
        for i, h5_file in enumerate(input_files):
            with prep_nisar.h5py.File(h5_file, "r") as ds:
                src_xcoord = ds[prep_nisar.DATASETS["xcoord"]][:]
                src_ycoord = ds[prep_nisar.DATASETS["ycoord"]][:]
                unw_data = ds[prep_nisar.DATASETS["unw"]][:]
                cor_data = ds[prep_nisar.DATASETS["cor"]][:]
                conn_comp = ds[prep_nisar.DATASETS["connComp"]][:]
                pbase = float(prep_nisar.np.nanmean(ds[prep_nisar.PROCESSINFO["bperp"]][()]))

            unw_resampled = resample_to_common_grid(
                unw_data,
                src_xcoord,
                src_ycoord,
                xcoord_common,
                ycoord_common,
                method="nearest",
            ).astype(prep_nisar.np.float32)
            cor_resampled = resample_to_common_grid(
                cor_data,
                src_xcoord,
                src_ycoord,
                xcoord_common,
                ycoord_common,
                method="nearest",
            ).astype(prep_nisar.np.float32)
            conn_resampled = resample_to_common_grid(
                conn_comp.astype(prep_nisar.np.float32),
                src_xcoord,
                src_ycoord,
                xcoord_common,
                ycoord_common,
                method="nearest",
            ).astype(prep_nisar.np.float32)
            conn_resampled[conn_resampled > 254] = prep_nisar.np.nan

            f["unwrapPhase"][i] = unw_resampled
            f["coherence"][i] = cor_resampled
            f["connectComponent"][i] = conn_resampled
            f["bperp"][i] = pbase
            prog_bar.update(i + 1, suffix=date12_list[i])
        prog_bar.close()

    print(f"finished writing to HDF5 file: {outfile}")
    return outfile


def read_and_interpolate_geometry_on_grid(
    meta_file,
    dem_file,
    xcoord_common,
    ycoord_common,
    mask_file=None,
):
    dem_dataset = gdal.Open(dem_file, gdal.GA_ReadOnly)
    proj = gdal.osr.SpatialReference(wkt=dem_dataset.GetProjection())
    dem_src_epsg = int(proj.GetAttrValue("AUTHORITY", 1))
    del dem_dataset

    rdr_coords = {}

    with prep_nisar.h5py.File(meta_file, "r") as ds:
        dst_epsg = ds[prep_nisar.DATASETS["epsg"]][()]
        rdr_coords["xcoord_radar_grid"] = ds[prep_nisar.PROCESSINFO["rdr_xcoord"]][()]
        rdr_coords["ycoord_radar_grid"] = ds[prep_nisar.PROCESSINFO["rdr_ycoord"]][()]
        rdr_coords["height_radar_grid"] = ds[prep_nisar.PROCESSINFO["rdr_height"]][()]
        rdr_coords["slant_range"] = ds[prep_nisar.PROCESSINFO["rdr_slant_range"]][()]
        rdr_coords["perp_baseline"] = ds[prep_nisar.PROCESSINFO["bperp"]][()]
        rdr_coords["incidence_angle"] = ds[prep_nisar.PROCESSINFO["rdr_incidence"]][()]

    subset_rows = len(ycoord_common)
    subset_cols = len(xcoord_common)
    y_2d, x_2d = prep_nisar.np.meshgrid(ycoord_common, xcoord_common, indexing="ij")
    bounds = (
        float(xcoord_common[0]),
        float(ycoord_common[-1]),
        float(xcoord_common[-1]),
        float(ycoord_common[0]),
    )
    output_projection = f"EPSG:{dst_epsg}"
    input_projection = f"EPSG:{dem_src_epsg}"

    with tempfile.TemporaryDirectory(prefix="mintpy_nisar_") as tmp_dir:
        output_dem = os.path.join(tmp_dir, "dem_transformed.tif")
        gdal.Warp(
            output_dem,
            dem_file,
            outputBounds=bounds,
            format="GTiff",
            srcSRS=input_projection,
            dstSRS=output_projection,
            resampleAlg=gdal.GRA_Bilinear,
            width=subset_cols,
            height=subset_rows,
            creationOptions=["COMPRESS=DEFLATE"],
        )
        dem_subset_array = gdal.Open(output_dem, gdal.GA_ReadOnly).ReadAsArray()

        slant_range, incidence_angle = interpolate_geometry(
            x_2d, y_2d, dem_subset_array, rdr_coords
        )

        if mask_file in ["auto", "None", None]:
            print("*** No mask was found ***")
            mask_subset_array = prep_nisar.np.ones(dem_subset_array.shape, dtype="byte")
        else:
            mask_dataset = gdal.Open(mask_file, gdal.GA_ReadOnly)
            proj = gdal.osr.SpatialReference(wkt=mask_dataset.GetProjection())
            mask_src_epsg = int(proj.GetAttrValue("AUTHORITY", 1))
            del mask_dataset

            output_mask = os.path.join(tmp_dir, "mask_transformed.tif")
            gdal.Warp(
                output_mask,
                mask_file,
                outputBounds=bounds,
                format="GTiff",
                srcSRS=f"EPSG:{mask_src_epsg}",
                dstSRS=output_projection,
                resampleAlg=gdal.GRA_NearestNeighbour,
                width=subset_cols,
                height=subset_rows,
                creationOptions=["COMPRESS=DEFLATE"],
            )
            mask_subset_array = gdal.Open(output_mask, gdal.GA_ReadOnly).ReadAsArray()

    return dem_subset_array, slant_range, incidence_angle, mask_subset_array


def interpolate_geometry(x_2d, y_2d, dem, rdr_coords):
    points = prep_nisar.np.stack((dem.flatten(), y_2d.flatten(), x_2d.flatten()), axis=-1)
    length, width = y_2d.shape
    interpolator = prep_nisar.RegularGridInterpolator((
                                                                                            rdr_coords['height_radar_grid'],
                                                                                            rdr_coords["ycoord_radar_grid"],
                                                                                            rdr_coords["xcoord_radar_grid"],),
                                                                                            rdr_coords["slant_range"],
                                                                                            method="linear",
                                                                                            bounds_error=False,
                                                                                            fill_value=None,
    )
    interpolated_slant_range = interpolator(points)

    interpolator = prep_nisar.RegularGridInterpolator(
        (
            rdr_coords['height_radar_grid'],
            rdr_coords["ycoord_radar_grid"],
            rdr_coords["xcoord_radar_grid"],
        ),
        rdr_coords["incidence_angle"],
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    interpolated_incidence_angle = interpolator(points)

    return (
        interpolated_slant_range.reshape(length, width),
        interpolated_incidence_angle.reshape(length, width),
    )


def main(iargs: list[str] | None = None) -> int:
    # Current ISCE3 GUNW products store radar-grid slant range as separate
    # reference/secondary datasets instead of the older generic slantRange name.
    prep_nisar.PROCESSINFO['rdr_slant_range'] = (
        "science/LSAR/GUNW/metadata/radarGrid/referenceSlantRange"
    )
    inps = cmd_line_parse(iargs)
    input_files = sorted(glob.glob(inps.input_glob))
    if len(input_files) == 0:
        raise FileNotFoundError(f"No GUNW files matched: {inps.input_glob}")

    bbox = None
    if inps.subset_lat:
        bbox = (inps.subset_lon[0], inps.subset_lat[0], inps.subset_lon[1], inps.subset_lat[1])

    metadata, xcoord_common, ycoord_common = extract_common_grid_metadata(input_files, bbox=bbox)
    stack_file = os.path.join(inps.out_dir, "inputs/ifgramStack.h5")
    geometry_file = os.path.join(inps.out_dir, "inputs/geometryGeo.h5")
    date12_list = get_date_pairs(input_files)

    metadata = prepare_geometry_common_grid(
        outfile=geometry_file,
        meta_file=input_files[0],
        metadata=metadata,
        dem_file=inps.dem_file,
        mask_file=inps.mask_file,
        xcoord_common=xcoord_common,
        ycoord_common=ycoord_common,
    )
    prepare_stack_common_grid(
        outfile=stack_file,
        input_files=input_files,
        metadata=metadata,
        date12_list=date12_list,
        xcoord_common=xcoord_common,
        ycoord_common=ycoord_common,
    )
    return 0



if __name__ == "__main__":
    main(sys.argv[1:])

    print("正在裁剪无效区域》》》》")

    # HDF5 文件路径
    ifg_stack_file = 'inputs/ifgramStack.h5'

    with h5py.File(ifg_stack_file, 'r+') as f:
        # 读取 unwrapPhase 和 coherence
        unwrap = f['unwrapPhase'][:]  # shape: (num_ifg, height, width)
        coh = f['coherence'][:]  # shape: (num_ifg, height, width)
        con=f['connectComponent'][:]
        bperp = f['bperp'][:]

        # -----------------------------
        # 将 unwrapPhase 的 0 值先设为 NaN
        # -----------------------------
        unwrap[unwrap == 0] = np.nan
        coh[coh == 0] = np.nan


        # -----------------------------
        # 生成无效像素 mask
        # 逻辑：只要某个图层该像素是 NaN 或 coherence=0，则该像素在所有图层都设为 NaN
        # -----------------------------
        # unwrapPhase 为 NaN 或 coherence 为 0
        invalid_mask = np.any(np.isnan(unwrap) , axis=0)  # shape: (height, width)

        # -----------------------------
        # 将无效像素在所有图层置为 NaN
        # -----------------------------
        unwrap[:, invalid_mask] = np.nan
        coh[:, invalid_mask] = np.nan
        con[:, invalid_mask]=np.nan
        ####去除unw的nan的行或列
        # 1. 找到所有非NaN数据的行列位置（仅用于确定边界）
        non_nan_mask = ~np.isnan(unwrap).all(axis=0)  # 任意干涉图有数据的像素
        rows_with_data, cols_with_data = np.where(non_nan_mask)

        # 2. 确定四周边界（仅调整到最近的有数据位置，内部行/列全部保留）
        # 第一行：所有有数据行的最小索引（不向内搜索，直接取边界）
        r_min = rows_with_data.min()
        # 最后一行：所有有数据行的最大索引（内部行/列完全保留）
        r_max = rows_with_data.max()
        # 第一列：所有有数据列的最小索引
        c_min = cols_with_data.min()
        # 最后一列：所有有数据列的最大索引
        c_max = cols_with_data.max()

        # 3. 截取范围（仅切四周边界，内部行/列全部保留，包括全NaN的）
        unwrap = unwrap[:, r_min:r_max + 1, c_min:c_max + 1]
        coh = coh[:, r_min:r_max + 1, c_min:c_max + 1]
        con = con[:, r_min:r_max + 1, c_min:c_max + 1]

        # 自动同步更新信息（只改这 5 个）
        n_rows, n_cols = unwrap.shape[1], unwrap.shape[2]
        ###获取X_FIRST、Y_FIRST、X_STEP、Y_STEP
        X_FIRST = f.attrs['X_FIRST']
        Y_FIRST = f.attrs['Y_FIRST']
        X_STEP = f.attrs['X_STEP']
        Y_STEP = f.attrs['Y_STEP']
        NEW_LENGTH = n_rows
        NEW_WIDTH = n_cols
        NEW_X_FIRST = float(X_FIRST) + float(c_min) * float(X_STEP)
        NEW_Y_FIRST = float(Y_FIRST) + float(r_min) * float(Y_STEP)

        # 新 bbox（纯数字计算，不拼字符串）
        x1 = float(NEW_X_FIRST)
        y1 = float(NEW_Y_FIRST)
        x2 = float(NEW_X_FIRST) + float(NEW_WIDTH - 1) * float(X_STEP)
        y2 = float(NEW_Y_FIRST) + float(NEW_LENGTH - 1) * float(Y_STEP)
        NEW_BBOX = f"{x1},{y2},{x2},{y1}"

        # -----------------------------
        # 保存回 HDF5
        # -----------------------------
        del f['unwrapPhase']
        f.create_dataset('unwrapPhase', data=unwrap)
        del f['coherence']
        f.create_dataset('coherence', data=coh)
        del f['connectComponent']
        f.create_dataset('connectComponent', data=con)
        f.attrs['LENGTH'] =str(NEW_LENGTH)
        f.attrs['WIDTH'] = str(NEW_WIDTH)
        f.attrs['X_FIRST'] = str(NEW_X_FIRST)
        f.attrs['Y_FIRST'] = str(NEW_Y_FIRST)
        f.attrs['bbox'] = str(NEW_BBOX)


        ##读取inputs/geometryGeo.h5，参照unwrap进行裁剪
        geo_file="./inputs/geometryGeo.h5"
    with h5py.File(geo_file, 'r+') as f:
        # 1. 读取数据
        height = f['height'][:]
        incidenceAngle = f['incidenceAngle'][:]
        slantRangeDistance = f['slantRangeDistance'][:]

        # 2. 裁剪（和上面保持一致）
        height = height[r_min:r_max + 1, c_min:c_max + 1]
        incidenceAngle = incidenceAngle[r_min:r_max + 1, c_min:c_max + 1]
        slantRangeDistance = slantRangeDistance[r_min:r_max + 1, c_min:c_max + 1]
        # 3. 删除旧数据，写入新裁剪后的数据
        del f['height']
        f.create_dataset('height', data=height)

        del f['incidenceAngle']
        f.create_dataset('incidenceAngle', data=incidenceAngle)

        del f['slantRangeDistance']
        f.create_dataset('slantRangeDistance', data=slantRangeDistance)

        # 4. 同步更新地理信息（和上面完全一样！）
        # 读取原始属性
        X_FIRST = f.attrs['X_FIRST']
        Y_FIRST = f.attrs['Y_FIRST']
        X_STEP = f.attrs['X_STEP']
        Y_STEP = f.attrs['Y_STEP']

        # 计算新值
        NEW_LENGTH = height.shape[0]
        NEW_WIDTH = height.shape[1]
        NEW_X_FIRST = float(X_FIRST) + float(c_min) * float(X_STEP)
        NEW_Y_FIRST = float(Y_FIRST) + float(r_min) * float(Y_STEP)

        x1 = NEW_X_FIRST
        y1 = NEW_Y_FIRST
        x2 = float(NEW_X_FIRST) + float(NEW_WIDTH - 1) * float(X_STEP)
        y2 = float(NEW_Y_FIRST) + float(NEW_LENGTH - 1) * float(Y_STEP)
        NEW_BBOX = f"{x1},{y2},{x2},{y1}"

        # 5. 写入更新后的属性
        f.attrs['LENGTH'] = str(NEW_LENGTH)
        f.attrs['WIDTH'] = str(NEW_WIDTH)
        f.attrs['X_FIRST'] = str(NEW_X_FIRST)
        f.attrs['Y_FIRST'] = str(NEW_Y_FIRST)
        f.attrs['bbox'] = str(NEW_BBOX)
        print("无效区域裁剪完成 ！！！")








