"""isce3 processing."""
import multiprocessing
import os
import re
import shutil
from datetime import datetime
import argparse
import glob
from pathlib import Path
from nisar.workflows import h5_prep, insar
from nisar.workflows.insar_runconfig import InsarRunConfig
from threading import Thread
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
                newstring += line.replace('sas_output_dir', out_path+"/GUNW_"+ date0 + "_" + date1+".h5")
            else:
                newstring = line
            yaml.write(newstring)

    return Path(out_path+'/insar.yaml')

def process_isce3(reference_path:str, secondary_path: str,dem_path:str) -> Path:
    yaml_path = get_config(reference_path, secondary_path,dem_path)
    args = argparse.Namespace(run_config_path=str(yaml_path), log_file=True)
    insar_runcfg = InsarRunConfig(args)

    run_steps = {
        'bandpass_insar': True,
        'rdr2geo': True,
        'geo2rdr': True,
        'prepare_insar_hdf5': True,
        'coarse_resample': True,
        'dense_offsets': False,
        'offsets_product': False,
        'rubbersheet': True,
        'fine_resample': True,
        'crossmul': True,
        'filter_interferogram': True,
        'unwrap': True,
        'ionosphere': True,
        'geocode': True,
        'solid_earth_tides': False,
        'baseline': True,
        'troposphere': True,
    }

    _, out_paths = h5_prep.get_products_and_paths(insar_runcfg.cfg)

    insar.run(cfg=insar_runcfg.cfg, out_paths=out_paths, run_steps=run_steps)

def make_sbas(n):
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

    # ===== Step 3: K近邻构网 =====
    pairs_set = set()

    for i, master in enumerate(files_sorted):
        # 计算与其它影像的时间差
        time_diffs = []
        for j, slave in enumerate(files_sorted):
            if i == j:
                continue
            dt = abs((dates[i] - dates[j]).total_seconds())
            time_diffs.append((dt, j, slave))

        # 按时间差排序，取最近 n 个
        time_diffs.sort(key=lambda x: x[0])
        nearest = time_diffs[:n]

        # 构建干涉对（去重：小时间在前）
        for _, j, slave in nearest:
            pair = tuple(sorted([master, slave], key=extract_date))
            pairs_set.add(pair)

    # ===== 转为列表并排序（便于查看）=====
    pairs = sorted(list(pairs_set), key=lambda x: extract_date(x[0]))
    #for i  in range(len(pairs)):
        #print(pairs[i][0], ",",pairs[i][1])
    return pairs

def mutl_run(data):
    ###获取当前路径
    fold_path = os.getcwd()
    reference_path = fold_path + "/rslc/" + str(data[0])
    secondary_path = fold_path + "/rslc/" + str(data[1])
    dem_path = fold_path + "/dem/dem.tif"
    ##获取data[0]、data[1]的日期
    data0 = Path(reference_path).name
    data1 = Path(secondary_path).name
    date0 = re.findall(r'\d{8}', str(data0))[0]
    date1 = re.findall(r'\d{8}', str(data1))[0]
    out_path = str(os.getcwd()) + "/isce3_process_data/" + date0 + "_" + date1


    gunw_file = str(os.getcwd()) +"/gunw/GUNW_"+ date0 + "_" + date1 + ".h5"
    if os.path.exists(gunw_file):
        print("GUNW_"+ date0 + "_" + date1 + ".h5已存在！")
    else:
        process_isce3(reference_path, secondary_path, dem_path)
        try:
            # 源文件
            src = out_path+"/GUNW_"+ date0 + "_" + date1+".h5"
            # 目标路径
            dst = str(os.getcwd()) +"/gunw/GUNW_"+ date0 + "_" + date1 + ".h5"
            # 移动
            shutil.move(src, dst)
            #删除过程文件
            shutil.rmtree(out_path)
        except:
            pass

if __name__ == '__main__':
    # 1. 定义命令行参数
    parser = argparse.ArgumentParser(description="NISAR4SBAS 并行预处理")
    parser.add_argument("-n", type=int, default=3, help="SBAS 相邻影像连接数量，默认3")
    parser.add_argument("-p", type=int, default=4, help="进程数，默认 4")
    args = parser.parse_args()
    # 2. 获取输入的 n 和 p
    n = args.n
    p = args.p
    data=make_sbas(n)
    try:
        os.mkdir("gunw")
    except:
        pass

    pool = multiprocessing.Pool(p)
    pool.map(mutl_run, data)
    pool.close()
    pool.join()

    try:
        shutil.rmtree("isce3_process_data")
    except:
        pass
    os.system("python ../nisar_code/prep_gunw.py -i './gunw/*.h5' -d './dem/dem.tif'")
    print("NISAR时序预处理已完成！！！")
    print("NISAR时序预处理已完成！！！")
    print("NISAR时序预处理已完成！！！")
