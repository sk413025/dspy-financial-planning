"""
Demo of the comprehensive logging system
"""
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.visualizer import ExperimentVisualizer

def demo_logging_system():
    """Demonstrate the logging capabilities"""
    
    print("=== DSPy 完整記錄系統展示 ===\n")
    
    # 1. 查看最新的日誌檔案
    logs_dir = Path("logs")
    if logs_dir.exists():
        session_files = list(logs_dir.glob("session_*.jsonl"))
        if session_files:
            latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 最新的 session 檔案: {latest_session}")
            
            # 讀取一個範例條目
            with open(latest_session, 'r') as f:
                lines = f.readlines()
                if lines:
                    entry = json.loads(lines[-1])  # 最後一個條目
                    
                    print("\n📋 記錄的資料結構:")
                    print("-" * 50)
                    
                    print("1. 基本資訊:")
                    print(f"   - ID: {entry['id']}")
                    print(f"   - 時間戳: {entry['timestamp']}")
                    print(f"   - 原始查詢: {entry['input']['raw_query']}")
                    print(f"   - 處理時間: {entry['duration_ms']:.0f}ms")
                    
                    print("\n2. 解析階段:")
                    if 'parsing' in entry['intermediate']:
                        parsing = entry['intermediate']['parsing']
                        print(f"   - 解析時間: {parsing['timestamp']}")
                        print("   - 解析參數:")
                        for key, value in parsing['parsed_params'].items():
                            print(f"     • {key}: {value}")
                    
                    print("\n3. 蒙特卡洛模擬:")
                    if 'monte_carlo' in entry['intermediate']:
                        mc = entry['intermediate']['monte_carlo']
                        print(f"   - 開始時間: {mc['start_timestamp']}")
                        print(f"   - 結束時間: {mc['end_timestamp']}")
                        if 'results' in mc:
                            results = mc['results']
                            print(f"   - 破產機率: {results['bankruptcy_prob']:.2f}%")
                            print(f"   - 模擬次數: {results['n_simulations']:,}")
                            print("   - 最終餘額統計:")
                            stats = results['final_balance_stats']
                            print(f"     • 平均: TWD {stats['mean']:,.0f}")
                            print(f"     • 中位數: TWD {stats['median']:,.0f}")
                            print(f"     • 標準差: TWD {stats['std']:,.0f}")
                        
                        # 檢查是否有路徑資料
                        if 'sample_paths' in mc:
                            n_paths = len(mc['sample_paths'])
                            path_length = len(mc['sample_paths'][0]) if n_paths > 0 else 0
                            print(f"   - 儲存的樣本路徑: {n_paths} 條，每條 {path_length} 年")
                    
                    print("\n4. 最終結果:")
                    if entry['output']:
                        output = entry['output']
                        print(f"   - 破產機率: {output['bankruptcy_probability']:.2f}%")
                        print(f"   - 符合目標: {'是' if output['meets_goal'] else '否'}")
                        print(f"   - 平均最終餘額: TWD {output['final_balance_mean']:,.0f}")
                    
                    print("\n5. 錯誤記錄:")
                    if entry['errors']:
                        for error in entry['errors']:
                            print(f"   - [{error['phase']}] {error['error']}")
                    else:
                        print("   - 無錯誤")
    
    print("\n\n=== 日誌檔案格式 ===")
    print("每個查詢都會產生一個完整的記錄，包含：")
    print("• 📝 JSONL 格式的詳細日誌 (session_YYYYMMDD_HHMMSS.jsonl)")
    print("• 📊 JSON 格式的摘要統計 (summary_YYYYMMDD_HHMMSS.json)")
    print("• 📈 可視化報告 (使用 visualizer.py 生成)")
    
    print("\n\n=== 可視化功能 ===")
    print("使用 ExperimentVisualizer 可以生成：")
    print("1. 蒙特卡洛路徑圖")
    print("2. 破產機率分布")
    print("3. 參數影響分析")
    print("4. 處理時間統計")
    print("5. 成功率分析")


def demo_analysis():
    """展示分析功能"""
    print("\n\n=== 資料分析展示 ===")
    
    # 獲取 logger 和資料
    logger = get_logger()
    df = logger.get_dataframe()
    
    if len(df) > 0:
        print(f"\n📊 目前 session 共有 {len(df)} 個查詢")
        
        # 基本統計
        print("\n基本統計:")
        print(f"- 平均處理時間: {df['duration_ms'].mean():.0f}ms")
        print(f"- 成功率: {df['success'].mean()*100:.1f}%")
        
        if 'bankruptcy_prob' in df.columns:
            print(f"- 平均破產機率: {df['bankruptcy_prob'].mean():.1f}%")
            print(f"- 破產機率範圍: {df['bankruptcy_prob'].min():.1f}% - {df['bankruptcy_prob'].max():.1f}%")
        
        # 參數分析
        print("\n參數統計:")
        param_cols = [col for col in df.columns if col.startswith('param_')]
        for col in param_cols:
            param_name = col.replace('param_', '')
            print(f"- {param_name}: {df[col].mean():.1f} (平均)")
    else:
        print("目前沒有資料可以分析")


def demo_visualization():
    """展示視覺化功能"""
    print("\n\n=== 視覺化功能展示 ===")
    
    # 檢查是否有資料可視覺化
    logs_dir = Path("logs")
    session_files = list(logs_dir.glob("session_*.jsonl"))
    
    if session_files:
        print("可用的視覺化功能:")
        print("1. python -m utils.visualizer --session-id SESSION_ID")
        print("2. 或在程式中使用:")
        print("""
from utils.visualizer import ExperimentVisualizer

# 建立視覺化器
viz = ExperimentVisualizer()

# 生成完整報告
viz.create_session_report()

# 或產生個別圖表
viz.plot_bankruptcy_distribution()
viz.plot_parameter_analysis()
viz.plot_monte_carlo_paths(entry_id=1)
        """)
        
        print("\n產生的視覺化內容包括:")
        print("• 🎯 破產機率分布圖")
        print("• 📈 蒙特卡洛路徑圖")
        print("• 🔍 參數影響分析")
        print("• ⏱️ 處理時間統計")
        print("• 📋 完整的 Markdown 報告")
    else:
        print("尚未有日誌資料，請先執行一些查詢")


if __name__ == "__main__":
    demo_logging_system()
    demo_analysis()
    demo_visualization()
    
    print("\n\n=== 總結 ===")
    print("完整的記錄系統提供：")
    print("✅ 自動記錄所有輸入、中間結果、最終輸出")
    print("✅ 詳細的時間戳和處理時間追蹤")
    print("✅ 錯誤記錄和除錯資訊")
    print("✅ 蒙特卡洛路徑和統計資料儲存")
    print("✅ 資料分析和視覺化工具")
    print("✅ 可讀的報告生成")
    
    print("\n這讓您可以：")
    print("• 追蹤 dspy 的解析效果")
    print("• 分析不同參數對結果的影響") 
    print("• 優化查詢和提示")
    print("• 生成專業的分析報告")