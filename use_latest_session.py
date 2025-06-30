"""
使用最新的 session 生成視覺化
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.visualizer import ExperimentVisualizer

def find_latest_session():
    """找到最新的 session"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return None
    
    session_files = list(logs_dir.glob("session_*.jsonl"))
    if not session_files:
        return None
    
    # 找到最新的文件
    latest_file = max(session_files, key=lambda x: x.stat().st_mtime)
    session_id = latest_file.stem.replace("session_", "")
    return session_id

def main():
    print("=== 使用最新 Session 的 ExperimentVisualizer ===\n")
    
    # 找到最新的 session
    latest_session_id = find_latest_session()
    
    if not latest_session_id:
        print("❌ 沒有找到任何 session 檔案")
        return
    
    print(f"🔍 找到最新 session: {latest_session_id}")
    
    # 使用最新的 session 創建視覺化器
    viz = ExperimentVisualizer(session_id=latest_session_id)
    
    print(f"📁 資料來源: {viz.session_file}")
    print(f"📊 找到 {len(viz.entries)} 個查詢記錄\n")
    
    if not viz.entries:
        print("❌ Session 檔案為空")
        return
    
    # 顯示查詢記錄概要
    print("📋 查詢記錄概要:")
    for i, entry in enumerate(viz.entries, 1):
        query = entry['input']['raw_query']
        if len(query) > 50:
            query = query[:47] + "..."
        
        duration = entry.get('duration_ms', 0)
        success = "✅" if not entry['errors'] else "❌"
        
        # 獲取破產機率
        bankruptcy_prob = "N/A"
        if entry.get('output') and 'bankruptcy_probability' in entry['output']:
            bankruptcy_prob = f"{entry['output']['bankruptcy_probability']:.1f}%"
        
        print(f"  {i}. {success} [{duration:.0f}ms] 破產率:{bankruptcy_prob} - {query}")
    
    print("\n" + "="*80)
    
    try:
        # 1. 生成完整報告
        print("\n📋 生成完整視覺化報告...")
        report_dir = viz.create_session_report()
        print(f"✅ 完整報告已生成到: {report_dir}")
        
        # 列出生成的檔案
        if Path(report_dir).exists():
            files = list(Path(report_dir).glob("*"))
            print(f"\n📁 生成了 {len(files)} 個檔案:")
            for file in sorted(files):
                file_size = file.stat().st_size
                if file_size > 1024:
                    size_str = f"{file_size/1024:.1f}KB"
                else:
                    size_str = f"{file_size}B"
                print(f"  📄 {file.name} ({size_str})")
        
        print(f"\n🔗 在瀏覽器中打開 {report_dir}/summary.md 查看完整報告")
        
    except Exception as e:
        print(f"❌ 生成報告時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 分別生成各種圖表
    print("\n" + "="*80)
    print("\n📈 個別生成視覺化圖表...")
    
    try:
        # 破產機率分布
        print("\n1. 生成破產機率分布圖...")
        viz.plot_bankruptcy_distribution(save_path="logs/latest_bankruptcy_distribution.png")
        print("✅ 保存到: logs/latest_bankruptcy_distribution.png")
    except Exception as e:
        print(f"❌ 破產機率分布圖生成失敗: {e}")
    
    try:
        # 參數分析
        print("\n2. 生成參數影響分析圖...")
        viz.plot_parameter_analysis(save_path="logs/latest_parameter_analysis.png")
        print("✅ 保存到: logs/latest_parameter_analysis.png")
    except Exception as e:
        print(f"❌ 參數分析圖生成失敗: {e}")
    
    # 蒙特卡洛路徑圖
    print("\n3. 生成蒙特卡洛路徑圖...")
    for entry in viz.entries:
        entry_id = entry['id']
        try:
            viz.plot_monte_carlo_paths(
                entry_id=entry_id, 
                save_path=f"logs/latest_monte_carlo_query_{entry_id}.png",
                n_paths=50
            )
            print(f"✅ 查詢 {entry_id} 路徑圖保存到: logs/latest_monte_carlo_query_{entry_id}.png")
        except Exception as e:
            print(f"❌ 查詢 {entry_id} 路徑圖生成失敗: {e}")
    
    # 3. 分析結果摘要
    print("\n" + "="*80)
    print("\n📊 分析結果摘要:")
    
    successful = [e for e in viz.entries if not e['errors']]
    bankruptcy_probs = []
    
    for entry in successful:
        if entry.get('output') and 'bankruptcy_probability' in entry['output']:
            bankruptcy_probs.append(entry['output']['bankruptcy_probability'])
    
    if bankruptcy_probs:
        print(f"\n💰 破產機率分析:")
        print(f"  查詢數量: {len(bankruptcy_probs)}")
        print(f"  平均破產機率: {sum(bankruptcy_probs)/len(bankruptcy_probs):.1f}%")
        print(f"  最低破產機率: {min(bankruptcy_probs):.1f}%")
        print(f"  最高破產機率: {max(bankruptcy_probs):.1f}%")
        
        # 風險分類
        low_risk = sum(1 for p in bankruptcy_probs if p <= 5)
        medium_risk = sum(1 for p in bankruptcy_probs if 5 < p <= 20)
        high_risk = sum(1 for p in bankruptcy_probs if p > 20)
        
        print(f"\n🎯 風險分布:")
        print(f"  低風險 (≤5%): {low_risk} 個查詢 ({low_risk/len(bankruptcy_probs)*100:.1f}%)")
        print(f"  中風險 (5-20%): {medium_risk} 個查詢 ({medium_risk/len(bankruptcy_probs)*100:.1f}%)")
        print(f"  高風險 (>20%): {high_risk} 個查詢 ({high_risk/len(bankruptcy_probs)*100:.1f}%)")
    
    # 處理時間分析
    durations = [e['duration_ms'] for e in successful if e.get('duration_ms')]
    if durations:
        print(f"\n⏱️  效能分析:")
        print(f"  平均處理時間: {sum(durations)/len(durations):.0f}ms")
        print(f"  最快查詢: {min(durations):.0f}ms")
        print(f"  最慢查詢: {max(durations):.0f}ms")
    
    print(f"\n📋 使用說明:")
    print(f"1. 查看完整報告: 打開 {report_dir}/summary.md")
    print(f"2. 查看圖表: 檢查 logs/ 目錄中的 .png 檔案")
    print(f"3. 原始資料: {viz.session_file}")

if __name__ == "__main__":
    main()