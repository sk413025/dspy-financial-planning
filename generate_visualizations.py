"""
使用 ExperimentVisualizer 生成視覺化報告
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.visualizer import ExperimentVisualizer

def main():
    print("=== 使用 ExperimentVisualizer 生成視覺化報告 ===\n")
    
    # 創建視覺化器
    viz = ExperimentVisualizer()
    
    print(f"正在分析 session: {viz.logger.session_id}")
    print(f"資料來源: {viz.session_file}")
    print(f"找到 {len(viz.entries)} 個查詢記錄\n")
    
    if not viz.entries:
        print("❌ 沒有找到查詢記錄")
        print("請先執行一些查詢：")
        print("python pipeline/run.py \"您的退休規劃查詢\"")
        return
    
    # 顯示可用的查詢
    print("📋 可用的查詢記錄:")
    for i, entry in enumerate(viz.entries, 1):
        query = entry['input']['raw_query'][:60] + "..." if len(entry['input']['raw_query']) > 60 else entry['input']['raw_query']
        duration = entry.get('duration_ms', 0)
        success = "✅" if not entry['errors'] else "❌"
        print(f"  {i}. {success} [{duration:.0f}ms] {query}")
    
    print("\n" + "="*70)
    
    # 1. 生成破產機率分布圖
    print("\n📊 1. 生成破產機率分布圖...")
    try:
        viz.plot_bankruptcy_distribution(save_path="logs/bankruptcy_distribution.png")
        print("✅ 破產機率分布圖已保存到: logs/bankruptcy_distribution.png")
    except Exception as e:
        print(f"❌ 生成破產機率分布圖失敗: {e}")
    
    # 2. 生成參數影響分析圖
    print("\n🔍 2. 生成參數影響分析圖...")
    try:
        viz.plot_parameter_analysis(save_path="logs/parameter_analysis.png")
        print("✅ 參數影響分析圖已保存到: logs/parameter_analysis.png")
    except Exception as e:
        print(f"❌ 生成參數影響分析圖失敗: {e}")
    
    # 3. 為每個查詢生成蒙特卡洛路徑圖
    print("\n📈 3. 生成蒙特卡洛路徑圖...")
    for entry in viz.entries:
        entry_id = entry['id']
        try:
            viz.plot_monte_carlo_paths(
                entry_id=entry_id, 
                save_path=f"logs/monte_carlo_paths_query_{entry_id}.png",
                n_paths=100
            )
            print(f"✅ 查詢 {entry_id} 的蒙特卡洛路徑圖已保存")
        except Exception as e:
            print(f"❌ 查詢 {entry_id} 的蒙特卡洛路徑圖生成失敗: {e}")
    
    # 4. 生成完整的視覺化報告
    print("\n📋 4. 生成完整報告...")
    try:
        report_dir = viz.create_session_report()
        print(f"✅ 完整報告已生成到: {report_dir}")
        
        # 列出報告內容
        if Path(report_dir).exists():
            files = list(Path(report_dir).glob("*"))
            print("\n📁 報告內容:")
            for file in files:
                print(f"  - {file.name}")
    except Exception as e:
        print(f"❌ 生成完整報告失敗: {e}")
    
    # 5. 顯示如何查看特定查詢的詳細信息
    print("\n" + "="*70)
    print("\n🔧 進階使用方法:")
    print("\n1. 查看特定查詢的蒙特卡洛路徑:")
    print("   viz.plot_monte_carlo_paths(entry_id=1, n_paths=50)")
    
    print("\n2. 自定義保存路徑:")
    print("   viz.plot_bankruptcy_distribution('my_custom_path.png')")
    
    print("\n3. 生成報告到指定目錄:")
    print("   viz.create_session_report('my_report_directory')")
    
    print("\n4. 分析特定 session:")
    print("   viz = ExperimentVisualizer(session_id='20250630_181418')")
    
    # 6. 顯示統計信息
    print("\n📊 Session 統計:")
    successful = [e for e in viz.entries if not e['errors']]
    failed = [e for e in viz.entries if e['errors']]
    
    print(f"  總查詢數: {len(viz.entries)}")
    print(f"  成功: {len(successful)} ({len(successful)/len(viz.entries)*100:.1f}%)")
    print(f"  失敗: {len(failed)} ({len(failed)/len(viz.entries)*100:.1f}%)")
    
    if successful:
        durations = [e['duration_ms'] for e in successful if e.get('duration_ms')]
        if durations:
            print(f"  平均處理時間: {sum(durations)/len(durations):.0f}ms")
            print(f"  最快/最慢: {min(durations):.0f}ms / {max(durations):.0f}ms")
    
    # 7. 顯示破產機率統計
    bankruptcy_probs = []
    for entry in successful:
        if entry.get('output') and 'bankruptcy_probability' in entry['output']:
            bankruptcy_probs.append(entry['output']['bankruptcy_probability'])
    
    if bankruptcy_probs:
        print(f"\n💰 破產機率統計:")
        print(f"  平均破產機率: {sum(bankruptcy_probs)/len(bankruptcy_probs):.1f}%")
        print(f"  最低/最高: {min(bankruptcy_probs):.1f}% / {max(bankruptcy_probs):.1f}%")
        low_risk = sum(1 for p in bankruptcy_probs if p <= 5)
        print(f"  低風險查詢 (≤5%): {low_risk}/{len(bankruptcy_probs)} ({low_risk/len(bankruptcy_probs)*100:.1f}%)")


if __name__ == "__main__":
    main()