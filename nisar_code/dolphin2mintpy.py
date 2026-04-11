

import glob
import os
import numpy as np
import h5py
import rasterio
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
try:
    os.makedirs("./inputs", exist_ok=True)
except:
    pass
#########################################################################################
### 读取相干性图
corr_dir = "./interferograms/*.int.cor.tif"
corr_list = sorted(glob.glob(corr_dir))

date_list_list = []
for f in corr_list:
    filename = os.path.basename(f)
    date0, date1 = filename.split(".")[0].split("_")
    date_list_list.append([date0, date1])

date_list = np.array(date_list_list, dtype='S8')
print(date_list_list)

m = len(corr_list)

# 获取尺寸
with rasterio.open(corr_list[0]) as ds:
    l, w = ds.height, ds.width

########################################################################################
### 解缠 & 连通图路径
unw_dir = "./unwrapped/*.unw.tif"
conn_dir = "./unwrapped/*.unw.conncomp.tif"

unw_list = sorted(glob.glob(unw_dir))
conn_list = sorted(glob.glob(conn_dir))

########################################################################################
##### 读取垂直基线
baseline_file = "./coregistered_slc/info/baseline/*_*.txt"
baseline_list = glob.glob(baseline_file)

date_baseline = []
for fpath in baseline_list:
    filename = os.path.basename(fpath)
    date0, date1 = filename.split(".")[0].split("_")

    with open(fpath, "r") as f:
        baseline_value = str(f.read().strip())

    date_baseline.append([date1, baseline_value])

#print(date_baseline)

# 转 numpy
date_baseline = np.array(date_baseline)
date_list_list = np.array(date_list_list)
dates = date_baseline[:, 0]
baselines = date_baseline[:, 1].astype(np.float32)
baseline_dict = dict(zip(dates, baselines))

# 计算 bperp
bperp_list = []
for date0, date1 in date_list_list:
    bperp = baseline_dict[date1] - baseline_dict[date0]
    bperp_list.append(bperp)

bperp_list = np.array(bperp_list, dtype=np.float32)

########################################################################################
# 是否保留干涉对
dropIfgram = np.ones(m, dtype=np.bool_)
########################################################################################
# 写入 HDF5（逐景写入）
out_file = "./inputs/geometryRadar.h5"

with h5py.File(out_file, "w") as f:
    # ===== 元数据
    f.attrs["FILE_TYPE"] = "geometryRadar"
    f.attrs["WIDTH"] = w
    f.attrs["LENGTH"] = l
    f.attrs["INTERFEROGRAM_NUMBER"] = m

    height_dir="./coregistered_slc/info/rdr2geo/z.rdr"
    longtiude_dir="./coregistered_slc/info/rdr2geo/x.rdr"
    latitude_dir="./coregistered_slc/info/rdr2geo/y.rdr"
    incidenceAngle_dir="./coregistered_slc/info/incidenceAngle.tif"
    slantRange_dir="./coregistered_slc/info/slantRange.tif"
    with rasterio.open(height_dir) as ds:
         height = ds.read(1)
         print("写入高程信息")
    f.create_dataset("height", data=height)

    with rasterio.open(longtiude_dir) as ds:
        longitude = ds.read(1)
        print("写入经度信息")
        f.create_dataset("longitude", data=longitude)

    with rasterio.open(latitude_dir) as ds:
        latitude = ds.read(1)
        print("写入纬度信息")
    f.create_dataset("latitude", data=latitude)
    with rasterio.open(incidenceAngle_dir) as ds:
        incidenceAngle = ds.read(1)
        print("写入入射角信息")
    f.create_dataset("incidenceAngle", data=incidenceAngle)
    with rasterio.open(slantRange_dir) as ds:
        slantRange= ds.read(1)
        print("写入斜距信息")
    f.create_dataset("slantRangeDistance",data=slantRange)

print("geometryRadar.h5 已生成")
print("                                              ")



########################################################################################
# 写入 HDF5（逐景写入）
out_file = "./inputs/ifgramStack.h5"

with h5py.File(out_file, "w") as f:

    # ===== 元数据
    f.attrs["FILE_TYPE"] = "ifgramStack"
    f.attrs["WIDTH"] = w
    f.attrs["LENGTH"] = l
    f.attrs["PROCESSOR"] = "isce"
    f.attrs["PLATFORM"]="NISAR"
    f.attrs["WAVELENGTH"] = "0.24393202441008952"
    f.attrs["ALOOKS"] = "1"
    f.attrs["RLOOKS"] = "1"

    # ===== 基础数据
    f.create_dataset("date", data=date_list)
    f.create_dataset("bperp", data=bperp_list)
    f.create_dataset("dropIfgram", data=dropIfgram)

    # ===== 创建空数据集（关键）
    dset_unw = f.create_dataset(
        "unwrapPhase", shape=(m, l, w),
        dtype=np.float32,  chunks=(1, l, w)
    )

    dset_coh = f.create_dataset(
        "coherence", shape=(m, l, w),
        dtype=np.float32,  chunks=(1, l, w)
    )

    dset_conn = f.create_dataset(
        "connectComponent", shape=(m, l, w),
        dtype=np.int16,  chunks=(1, l, w)
    )

    # ======================
    # 写 coherence（逐景）
    # ======================
    for i, fpath in enumerate(corr_list):
        print(f"[{i+1}/{m}] 写入 coherence")
        with rasterio.open(fpath) as ds:
            data = ds.read(1)
            dset_coh[i, :, :] = data

    # ======================
    # 写 unwrapPhase
    # ======================
    for i, fpath in enumerate(unw_list):
        print(f"[{i+1}/{m}] 写入 unwrapPhase")
        with rasterio.open(fpath) as ds:
            dset_unw[i, :, :] = ds.read(1)

    # ======================
    # 写 connectComponent
    # ======================
    for i, fpath in enumerate(conn_list):
        print(f"[{i+1}/{m}] 写入 connectComponent")
        with rasterio.open(fpath) as ds:
            dset_conn[i, :, :] = ds.read(1)

print("ifgramStack.h5 已生成")
print("                                               ")

print("已完成Mintpy输入数据准备！！！")
print("已完成Mintpy输入数据准备！！！")
print("已完成Mintpy输入数据准备！！！")