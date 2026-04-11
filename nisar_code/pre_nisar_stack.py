import multiprocessing
import os
import re
import shutil
from datetime import datetime
import argparse
from pathlib import Path
from nisar.workflows import h5_prep
import insar
from nisar.workflows.insar_runconfig import InsarRunConfig
import h5py
import glob
from osgeo import gdal
import numpy as np
from scipy.ndimage import zoom
import tifffile as tiff


def get_config(reference_path: str, secondary_path: str,dem_path:str) -> Path:
    """Create a configuration file for isce3.

    Args:
        reference_path: Path of the reference scene.
        secondary_path: Path of the secondary scene.

    Returns:
        yaml_file: Path of the configuration file.
    """
    data0=Path(reference_path).name
    data1=Path(secondary_path).name
    date0 = re.findall(r'\d{8}', str(data0))[0]
    date1 = re.findall(r'\d{8}', str(data1))[0]

    out_path = str(os.getcwd()) + "/isce3_process_data/" + date0 + "_" + date1
    try:
        os.mkdir(str(os.getcwd()) + "/isce3_process_data")
    except:
        pass
    try:
        os.mkdir(out_path)
    except:
        pass
    yaml_schema =Path( "../nisar_code/schemas/insar.yaml")
    with yaml_schema.open('r') as yaml:
        lines = yaml.readlines()

    yaml_file = Path(out_path+'/insar.yaml')
    with yaml_file.open('w') as yaml:
        for line in lines:
            newstring = ''
            if 'reference_scene' in line:
                newstring += line.replace('reference_scene', reference_path)
            elif 'secondary_scene' in line:
                newstring += line.replace('secondary_scene', secondary_path)
            elif 'dem.tif' in line:
                newstring += line.replace('dem.tif', dem_path)
            elif 'product_dir:' in line:
                newstring += line.replace('product_dir', out_path)
            elif 'scratch_dir' in line:
                newstring+= line.replace('scratch_dir', out_path)
            elif 'sas_output_dir' in line:
                newstring += line.replace('sas_output_dir', out_path+"/GUNW.h5")
            else:
                newstring = line
            yaml.write(newstring)

    return Path(out_path+'/insar.yaml')



def process_isce3(reference_path:str, secondary_path: str,dem_path:str) -> Path:
    ##获取ref、sec的日期
    data0=Path(reference_path).name
    data1=Path(secondary_path).name
    date0 = re.findall(r'\d{8}', str(data0))[0]
    date1 = re.findall(r'\d{8}', str(data1))[0]

    ####判断某文件是否存在
    if os.path.exists("./coregistered_slc/"+date1+".slc") and os.path.exists("./coregistered_slc/" + date1 + ".hdr"):
        print("coregistered_slc文件夹下已存在"+str(date1)+"配准结果，跳过该数据配准步骤。\n")
    else:
        yaml_path = get_config(reference_path, secondary_path,dem_path)
        args = argparse.Namespace(run_config_path=str(yaml_path), log_file=False)
        insar_runcfg = InsarRunConfig(args)

        run_steps = {
            'bandpass_insar': True,
            'rdr2geo': True,
            'geo2rdr': True,
            'prepare_insar_hdf5': True,
            'coarse_resample': True,
            'dense_offsets': True,
            'offsets_product': True,
            'rubbersheet': True,
            'fine_resample': True,
            'crossmul': False,
            'filter_interferogram': False,
            'unwrap': False,
            'ionosphere': False,
            'geocode': False,
            'solid_earth_tides': False,
            'baseline': True,
            'troposphere': False
        }
        _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

        insar.run(cfg=insar_runcfg.cfg, out_paths=out_paths, run_steps=run_steps)

        out_path_slc =  "./isce3_process_data/" + date0 + "_" + date1+"/fine_resample_slc/freqA/HH/coregistered_secondary.slc"
        out_path_hdr = "./isce3_process_data/" + date0 + "_" + date1 + "/fine_resample_slc/freqA/HH/coregistered_secondary.hdr"
        try:
            shutil.copy2(out_path_slc, "./coregistered_slc/"+date1+".slc")
            shutil.copy2(out_path_hdr, "./coregistered_slc/" + date1 + ".hdr")
        except:
            pass

        GUNW = "./isce3_process_data/" + str(date0) + "_" + str(date1) + "/GUNW.h5"
        with h5py.File(GUNW, 'r') as f:
            perpendicularBaseline = f["/science/LSAR/GUNW/metadata/radarGrid/perpendicularBaseline"][0]
            final_perpendicularBaseline = np.nanmean(perpendicularBaseline)
            with open("./coregistered_slc/info/baseline/" + str(date0) + "_" + str(date1) + ".txt", "w") as f:
                f.write(str(final_perpendicularBaseline))


        try:

            if date0==date1:
                print("                   ")
            else:
                print("                   ")
                shutil.rmtree("./isce3_process_data/" + date0 + "_" + date1)
        except:
            pass



def make_single(reference_date):
    current_path = os.getcwd()
    #####获取当前路径以.h5为后缀的所有文件名
    files_path=glob.glob(str(current_path) + '/rslc/*.h5')
    files= [os.path.basename(f) for f in files_path]


    # ===== 提取时间 =====
    def extract_date(file):
        match = re.search(r'_(\d{8}T\d{6})_', file)
        if not match:
            raise ValueError(f"未找到日期: {file}")
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
        # ===== Step 1: 排序 =====

    files_sorted = sorted(files, key=extract_date)
    # ===== Step 2: 构建时间列表 =====
    dates = [extract_date(f) for f in files_sorted]
    data_list=[]
    for i in  range(len(files_sorted)):
        if reference_date == 0:
            reference_path =  "./rslc/" + files_sorted[0]
        elif reference_date == 1:
            reference_path= "./rslc/" + files_sorted[int(len(files_sorted)/2)]
        elif reference_date == 2:
            reference_path =  "./rslc/" + files_sorted[-1]
        secondary_path =  "./rslc/" + files_sorted[i]
        one_data=[reference_path, secondary_path]
        data_list.append(one_data)
    return data_list,dates[0]



def mutl_slc(data):

    dem_path="./dem/dem.tif"
    process_isce3(reference_path=data[0],secondary_path=data[1],dem_path=dem_path)







# =========================
# tiff裁剪
# =========================
def crop_tif(input_path, output_path=None):
    ds = gdal.Open(input_path)
    data = ds.GetRasterBand(1).ReadAsArray()

    # 尺寸检查
    if data.shape != mask.shape:
        print(f"跳过: {input_path}")
        return

    # 裁剪
    cropped = data[row_min:row_max+1, col_min:col_max+1]

    # 保存（覆盖或另存）
    if output_path is None:
        output_path = input_path

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path,
                           cropped.shape[1],
                           cropped.shape[0],
                           1,
                           gdal.GDT_Float32)

    out_ds.GetRasterBand(1).WriteArray(cropped)
    out_ds.FlushCache()

    print(f"已裁剪: {input_path}")



# =========================
# rdr与hdr裁剪
# =========================
def crop_rdr(input_rdr, output_rdr=None):
    # 打开数据（ENVI格式会自动读取hdr）
    ds = gdal.Open(input_rdr)
    data = ds.GetRasterBand(1).ReadAsArray()

    # 尺寸检查
    if data.shape != mask.shape:
        print(f"跳过: {input_rdr}")
        return
    # 裁剪
    cropped = data[row_min:row_max+1, col_min:col_max+1]
    # 输出路径
    if output_rdr is None:
        output_rdr = input_rdr
    # ENVI驱动（会自动生成hdr）
    driver = gdal.GetDriverByName("ENVI")
    out_ds = driver.Create(
        output_rdr,
        cropped.shape[1],   # 列数
        cropped.shape[0],   # 行数
        1,
        gdal.GDT_Float32
    )
    # 写数据
    out_ds.GetRasterBand(1).WriteArray(cropped)
    # 拷贝元数据（关键）
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"已裁剪: {input_rdr}")

def crop_rdr_slc(input_rdr, output_rdr=None):
    # 打开数据（ENVI格式会自动读取hdr）
    ds = gdal.Open(input_rdr)
    data = ds.GetRasterBand(1).ReadAsArray()

    # 尺寸检查
    if data.shape != mask.shape:
        print(f"跳过: {input_rdr}")
        return
    # 裁剪
    cropped = data[row_min:row_max+1, col_min:col_max+1]
    # 输出路径
    if output_rdr is None:
        output_rdr = input_rdr
    # ENVI驱动（会自动生成hdr）
    driver = gdal.GetDriverByName("ENVI")
    out_ds = driver.Create(
        output_rdr,
        cropped.shape[1],   # 列数
        cropped.shape[0],   # 行数
        1,
        gdal.GDT_CFloat32
    )
    # 写数据
    out_ds.GetRasterBand(1).WriteArray(cropped)
    # 拷贝元数据（关键）
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.FlushCache()
    out_ds = None
    print(f"已裁剪: {input_rdr}")





#Coregistered SLC
if  __name__ == "__main__":

    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="NISAR4DS并行预处理")
    parser.add_argument("-p", type=int, default=8, help="进程数，默认 8")
    args = parser.parse_args()
    # 2. 获取输入的p
    p = args.p
    try:
        os.mkdir(str(os.getcwd()) + "/isce3_process_data")
    except:
        pass
    try:
        os.mkdir(str(os.getcwd()) + "/coregistered_slc")
    except:
        pass

    try:
        os.mkdir(str(os.getcwd()) + "/coregistered_slc/info")
    except:
        pass
    try:
        os.mkdir(str(os.getcwd()) + "/coregistered_slc/info/baseline")
    except:
        pass



    data,ref_date= make_single(0)
    pool = multiprocessing.Pool(p)
    pool.map(mutl_slc, data)
    pool.close()
    pool.join()




    ###################################################################

    # 读取所有 SLC 文件
    slc_list = sorted(glob.glob("./coregistered_slc/*.slc"))
    mask_list = []
    for i in range(len(slc_list)):
        # 读取数据
        print("读取第"+str(i+1)+"个配准后的slc文件,计算实际可用区域。")
        ds = gdal.Open(slc_list[i])
        data = ds.GetRasterBand(1).ReadAsArray()
        # 生成非空mask（幅度不为0）
        mask = np.abs(data) > 0
        mask_list.append(mask)
    # 求所有mask的交集（公共非空区域）
    common_mask = np.logical_and.reduce(mask_list)
    #####保存mask为tiff文件
    ref_ds = gdal.Open(slc_list[0])
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize
    # 创建输出文件
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create("./coregistered_slc/info/slc_mask.tif", cols, rows, 1, gdal.GDT_Byte)
    # 写入数据（True/False → 1/0）
    print("生成slc掩膜文件 slc_mask.tif")
    out_ds.GetRasterBand(1).WriteArray(common_mask.astype(np.uint8))
    out_ds = None
    #################################################################



    ####获取某路径下包含data0的文件
    data0 = str(datetime.strptime(str(ref_date), "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d"))
    if os.path.exists("./isce3_process_data/"+str(data0)+"_"+str(data0)):
        rslc_list = glob.glob("./rslc/*" + str(data0) + "*_clipped.h5")
        RSLC = rslc_list[0]
        with h5py.File(RSLC, 'r') as f:
            slc_shape = f["/science/LSAR/RSLC/swaths/frequencyA/HH"][:].shape
            slant_range = f["/science/LSAR/RSLC/swaths/frequencyA/slantRange"][:]
            final_slant_range = np.broadcast_to(slant_range[None, :], (slc_shape[0], slc_shape[1]))
            ###保存为tiff文件
            print("生成slantRange.tif文件")
            tiff.imwrite("./coregistered_slc/info/slantRange.tif", final_slant_range)
        #######rifg
        RIFG = "./isce3_process_data/" + str(data0) + "_" + str(data0)+ "/RIFG.h5"
        with h5py.File(RIFG, 'r') as f:
            incidenceAngle = f["/science/LSAR/RIFG/metadata/geolocationGrid/incidenceAngle"][0]
        zoom_factors = (slc_shape[0] / incidenceAngle.shape[0], slc_shape[1] / incidenceAngle.shape[1])
        final_incidenceAngle = zoom(incidenceAngle, zoom_factors, order=1)
        print("生成incidenceAngle.tif文件")
        tiff.imwrite("./coregistered_slc/info/incidenceAngle.tif", final_incidenceAngle)
        #################################################################
        try:
            os.mkdir(str(os.getcwd()) + "/coregistered_slc/info/rdr2geo" )
        except:
            pass
        try:
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0)+"/rdr2geo/freqA/x.hdr","./coregistered_slc/info/rdr2geo/x.hdr")
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0)+ "/rdr2geo/freqA/x.rdr", "./coregistered_slc/info/rdr2geo/x.rdr")
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0) + "/rdr2geo/freqA/y.hdr","./coregistered_slc/info/rdr2geo/y.hdr")
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0) + "/rdr2geo/freqA/y.rdr","./coregistered_slc/info/rdr2geo/y.rdr")
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0) + "/rdr2geo/freqA/z.hdr","./coregistered_slc/info/rdr2geo/z.hdr")
            shutil.copy2("./isce3_process_data/" + str(data0) + "_" + str(data0) + "/rdr2geo/freqA/z.rdr","./coregistered_slc/info/rdr2geo/z.rdr")
        except:
            pass
        #################################################################
        try:
            shutil.rmtree("./isce3_process_data")
        except:
            pass
    #############裁剪所有数据至公共区###########################
    slc_mask=gdal.Open("./coregistered_slc/info/slc_mask.tif")
    mask_end=slc_mask.GetRasterBand(1).ReadAsArray()

    valid_rows = np.where(np.any(mask_end == 1, axis=1))[0]
    valid_cols = np.where(np.any(mask_end == 1, axis=0))[0]
    row_min, row_max = valid_rows[0], valid_rows[-1]
    col_min, col_max = valid_cols[0], valid_cols[-1]


    if row_min == 0 and row_max == mask_end.shape[0] - 1 and col_min==0 and col_max == mask_end.shape[1] - 1 :
        print("无需裁剪，所有数据已在公共区内。")
    else:
        for i in range(len(slc_list)):
            crop_rdr_slc(slc_list[i])
        geo_list=glob.glob("./coregistered_slc/info/rdr2geo/*.rdr")
        for i in range(len(geo_list)):
            crop_rdr(geo_list[i])
        tif_list=glob.glob("./coregistered_slc/info/*.tif")
        for i in range(len(tif_list)):
            crop_tif(tif_list[i])
    #################################################################



    print("NISAR时序配准流程已完成！！！")
    print("NISAR时序配准流程已完成！！！")
    print("NISAR时序配准流程已完成！！！")
