import os
import time
import shutil
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
# 基础路径
BASE_PATH = "/mnt/shared-storage-user/chenguanjie/huawei_eplb/data/domain_shift_hf/benchmarks"

# 需要下载的数据集列表
DATASETS_TO_DOWNLOAD = [
    "allenai/winogrande",
    "google/boolq",
    "tatsu-lab/alpaca",
    "hotpot_qa"
]

# 镜像站设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 重试设置
MAX_RETRIES = 3          # 每个数据集最多尝试几次
RETRY_DELAY = 15         # 失败后等待多少秒重试 (稍微延长一点)
# ===========================================

def get_local_dir_name(repo_id):
    """将 repo_id 转换为文件夹名"""
    return repo_id.replace("/", "_")

def download_dataset(repo_id, local_dir):
    print(f"\n{'='*60}")
    print(f"正在处理: {repo_id}")
    print(f"保存路径: {local_dir}")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[尝试 {attempt}/{MAX_RETRIES}] 开始下载...")
            
            # 简化参数，只保留最核心的
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                resume_download=True, # 断点续传
            )
            
            print(f"✅ [成功] {repo_id} 下载完成！")
            return True # 成功

        except Exception as e:
            print(f"❌ [错误] 下载发生异常: {str(e)}")
            
            if attempt < MAX_RETRIES:
                print(f"⏳ 等待 {RETRY_DELAY} 秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"💀 [失败] {repo_id} 已达到最大重试次数 ({MAX_RETRIES})。")
                return False # 失败

def main():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    
    failed_datasets = []

    for repo_id in DATASETS_TO_DOWNLOAD:
        folder_name = get_local_dir_name(repo_id)
        local_dir = os.path.join(BASE_PATH, folder_name)
        
        success = download_dataset(repo_id, local_dir)
        
        if not success:
            failed_datasets.append(repo_id)
            # 清理逻辑
            if os.path.exists(local_dir):
                print(f"🧹 [清理] 正在删除不完整的文件夹: {local_dir}")
                try:
                    shutil.rmtree(local_dir)
                except Exception as e:
                    print(f"⚠️  警告: 无法删除文件夹 {local_dir}: {e}")

    # 最终总结
    print(f"\n\n{'='*60}")
    print("全部任务结束。")
    if failed_datasets:
        print(f"以下数据集下载失败（已清理）:")
        for ds in failed_datasets:
            print(f"  - {ds}")
    else:
        print("🎉 所有数据集下载成功！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()