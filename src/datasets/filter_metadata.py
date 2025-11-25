import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

B_FACTOR_THRESHOLD = 80
CONFIDENCE_THRESHOLD = 90

def filter_samples(data_root, data_type, request_complex=False, output_file="filtered_samples.tsv"):
    """
    筛选满足bfactor条件的ESM和标签文件路径对
    
    参数:
        data_root: PDB数据根目录
        data_type: 数据类型 ('exp'或'pred')
        output_file: 输出文件名称
    """
    # 验证数据类型
    if data_type not in ['exp', 'pred']:
        raise ValueError("data_type必须是'exp'(实验数据)或'pred'(预测数据)")
    
    # 收集所有ESM编码文件路径
    esm_paths = glob.glob(os.path.join(data_root, "esm_encodings/*/*.npz")) + glob.glob(os.path.join(data_root, "esm_encodings/*.npz"))
    
    print(f"Finding {len(esm_paths)} ESM encoding npz files under directory {data_root}")
    print(f"Data type: {'Experimental PDB' if data_type == 'exp' else 'Predicted (AlphaFold)'}")
    
    # 筛选有效样本
    valid_samples = []
    skipped_no_label = 0
    skipped_condition = 0
    skipped_other = 0
    
    for esm_p in tqdm(esm_paths, desc="处理样本"):
        # try two naming conventions
        label_p1 = esm_p.replace('esm_encodings', 'labels')[:-4] + '.tsv'
        label_p2 = esm_p.replace('esm_encodings', 'labels')[:-5] + '.tsv'
        
        label_p = None
        for lp in [label_p1, label_p2]:
            if os.path.exists(lp):
                label_p = lp
                break
        
        if not label_p:
            skipped_no_label += 1
            continue  # 没有找到对应的标签文件
        
        # 检查样本是否满足条件
        conditions = ['bfactor' if data_type == 'exp' else 'confidence']
        if request_complex: conditions.append('is_complex')
        is_valid, reason = is_valid_sample(label_p, conditions)
        if is_valid:
            valid_samples.append((esm_p, label_p))
        else:
            print(reason)
            if "Condition not met" in reason:
                skipped_condition += 1
            else:
                skipped_other += 1
    
    # 保存结果
    output_path = os.path.join(data_root, output_file)
    with open(output_path, 'w') as f:
        f.write("esm_path\tlabel_path\n")
        for esm_p, label_p in valid_samples:
            f.write(f"{esm_p}\t{label_p}\n")
    
    print("\nFiltering Results:")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Skipped samples (no label file): {skipped_no_label}")
    print(f"Skipped samples (condition not met): {skipped_condition}")
    print(f"Skipped samples (other errors): {skipped_other}")
    print(f"Results saved to: {output_path}")

    threshold = B_FACTOR_THRESHOLD if data_type == 'exp' else CONFIDENCE_THRESHOLD
    condition_desc = "Average bfactor <" if data_type == 'exp' else "Average confidence >"
    print(f"Condition: {condition_desc} {threshold}")
    if request_complex:
        print("Additional condition: Must be a protein complex (at least 2 chains)")

def is_valid_sample(label_path, conditions=['bfactor']):
    """
    检查样本是否满足bfactor条件
    
    参数:
        label_path: 标签文件路径
        conditions: 筛选条件，bfactor\confidence\is_complex等

    返回:
        tuple: (是否有效, 原因说明)
    """
    try:
        # 读取标签文件
        df = pd.read_csv(label_path, sep='\t')
        
        # 检查必要的列是否存在
        if 'bfactor' not in df.columns:
            return False, f"'{label_path}'中缺少'bfactor'列"
        
        # 提取bfactor列
        bfactor_col = 'bfactor'
        bfactors = pd.to_numeric(df[bfactor_col], errors='coerce')
        
        # 移除NaN值
        valid_bfactors = bfactors.dropna()
        if len(valid_bfactors) == 0:
            return False, f"'{label_path}'中没有有效的bfactor值"
        
        # 计算平均bfactor
        mean_bfactor = valid_bfactors.mean()
        
        # 根据数据类型应用阈值
        if 'bfactor' in conditions:  # exp
            if mean_bfactor > B_FACTOR_THRESHOLD:
                return (False, f"Condition not met: Average bfactor {mean_bfactor:.2f} > B_FACTOR_THRESHOLD")
        if 'confidence' in conditions:  # pred
            if mean_bfactor < CONFIDENCE_THRESHOLD:
                return (False, f"Condition not met: Average confidence {mean_bfactor:.2f} < CONFIDENCE_THRESHOLD")
        if 'is_complex' in conditions:
            chains = df['Chain'].unique()
            if len(chains) < 2:
                return (False, f"Condition not met: Non-protein complex (Number of chains: {len(chains)})")
        return True, "Valid sample"

    except Exception as e:
        return False, f"Error processing '{label_path}': {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="筛选满足bfactor条件的ESM和标签文件路径对",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("data_root", 
                        help="包含esm_encodings和labels目录的数据根路径")
    parser.add_argument("--type", 
                        choices=['exp', 'pred'], 
                        required=True,
                        help="数据类型: 'exp'=实验PDB, 'pred'=预测(如AlphaFold)")
    parser.add_argument("--request_complex", 
                        action='store_true',
                        help="是否仅筛选蛋白复合物样本")
    parser.add_argument("--output", 
                        default="filtered_samples.tsv",
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    print(f"开始筛选数据目录: {args.data_root}")
    filter_samples(args.data_root, args.type, request_complex=args.request_complex, output_file=args.output)
