"""
使用所有 session 數據的 ExperimentVisualizer 示例
"""
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.visualizer import ExperimentVisualizer

def demo_all_sessions():
    """展示如何分析所有 session 的數據"""
    print("=== ExperimentVisualizer 完整功能展示 ===\n")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("❌ logs 目錄不存在")
        return
    
    # 找到所有 session 檔案
    session_files = list(logs_dir.glob("session_*.jsonl"))
    if not session_files:
        print("❌ 沒有找到任何 session 檔案")
        return
    
    print(f"🔍 找到 {len(session_files)} 個 session 檔案:")
    
    all_entries = []
    
    # 讀取所有 session 的數據
    for session_file in sorted(session_files):
        session_id = session_file.stem.replace("session_", "")
        print(f"  📁 {session_id}")
        
        try:
            with open(session_file, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entry['session_id'] = session_id  # 添加 session 標識
                        all_entries.append(entry)
        except Exception as e:
            print(f"    ❌ 讀取失敗: {e}")
    
    print(f"\n📊 總共找到 {len(all_entries)} 個查詢記錄\n")
    
    # 分析所有查詢
    print("📋 所有查詢記錄:")
    print("-" * 100)
    print(f"{'ID':<3} {'Session':<15} {'成功':<4} {'時間':<8} {'破產率':<8} {'查詢':<50}")
    print("-" * 100)
    
    for entry in all_entries:
        session_id = entry.get('session_id', 'Unknown')[:15]
        success = "✅" if not entry.get('errors', []) else "❌"
        duration = f"{entry.get('duration_ms', 0):.0f}ms"
        
        bankruptcy_prob = "N/A"
        if entry.get('output') and 'bankruptcy_probability' in entry['output']:
            bankruptcy_prob = f"{entry['output']['bankruptcy_probability']:.1f}%"
        
        query = entry['input']['raw_query']
        if len(query) > 45:
            query = query[:42] + "..."
        
        print(f"{entry['id']:<3} {session_id:<15} {success:<4} {duration:<8} {bankruptcy_prob:<8} {query:<50}")
    
    print("-" * 100)
    
    # 統計分析
    successful_entries = [e for e in all_entries if not e.get('errors', [])]
    failed_entries = [e for e in all_entries if e.get('errors', [])]
    
    print(f"\n📊 統計摘要:")
    print(f"  總查詢數: {len(all_entries)}")
    print(f"  成功查詢: {len(successful_entries)} ({len(successful_entries)/len(all_entries)*100:.1f}%)")
    print(f"  失敗查詢: {len(failed_entries)} ({len(failed_entries)/len(all_entries)*100:.1f}%)")
    
    # 破產機率分析
    bankruptcy_probs = []
    for entry in successful_entries:
        if entry.get('output') and 'bankruptcy_probability' in entry['output']:
            bankruptcy_probs.append(entry['output']['bankruptcy_probability'])
    
    if bankruptcy_probs:
        print(f"\n💰 破產機率分析:")
        print(f"  有效查詢數: {len(bankruptcy_probs)}")
        print(f"  平均破產機率: {sum(bankruptcy_probs)/len(bankruptcy_probs):.1f}%")
        print(f"  最低破產機率: {min(bankruptcy_probs):.1f}%")
        print(f"  最高破產機率: {max(bankruptcy_probs):.1f}%")
        
        # 風險分布
        low_risk = sum(1 for p in bankruptcy_probs if p <= 5)
        medium_risk = sum(1 for p in bankruptcy_probs if 5 < p <= 20)
        high_risk = sum(1 for p in bankruptcy_probs if p > 20)
        
        print(f"\n🎯 風險分布:")
        print(f"  低風險 (≤5%): {low_risk} ({low_risk/len(bankruptcy_probs)*100:.1f}%)")
        print(f"  中風險 (5-20%): {medium_risk} ({medium_risk/len(bankruptcy_probs)*100:.1f}%)")
        print(f"  高風險 (>20%): {high_risk} ({high_risk/len(bankruptcy_probs)*100:.1f}%)")
    
    # 處理時間分析
    durations = [e['duration_ms'] for e in successful_entries if e.get('duration_ms')]
    if durations:
        print(f"\n⏱️  效能分析:")
        print(f"  平均處理時間: {sum(durations)/len(durations):.0f}ms")
        print(f"  最快查詢: {min(durations):.0f}ms")
        print(f"  最慢查詢: {max(durations):.0f}ms")
    
    # 參數分析
    print(f"\n📈 參數分析:")
    param_stats = {}
    for entry in successful_entries:
        if 'parsing' in entry.get('intermediate', {}):
            params = entry['intermediate']['parsing']['parsed_params']
            for key, value in params.items():
                if key not in param_stats:
                    param_stats[key] = []
                param_stats[key].append(value)
    
    for param, values in param_stats.items():
        if param in ['yrs', 'return_mu', 'return_sigma', 'inflation', 'goal_pct']:
            avg_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            print(f"  {param}: 平均={avg_val:.1f}, 範圍={min_val:.1f}-{max_val:.1f}")
        elif param in ['spend', 'init_net']:
            avg_val = sum(values) / len(values) / 1e6  # 轉為百萬
            min_val = min(values) / 1e6
            max_val = max(values) / 1e6
            print(f"  {param}: 平均={avg_val:.1f}M, 範圍={min_val:.1f}M-{max_val:.1f}M TWD")


def demo_individual_visualizations():
    """展示個別視覺化功能"""
    print(f"\n\n=== 個別視覺化功能展示 ===\n")
    
    # 找到最新的 session
    logs_dir = Path("logs")
    session_files = list(logs_dir.glob("session_*.jsonl"))
    
    if not session_files:
        print("❌ 沒有 session 檔案可以視覺化")
        return
    
    latest_session = max(session_files, key=lambda x: x.stat().st_mtime)
    session_id = latest_session.stem.replace("session_", "")
    
    print(f"🎯 使用最新 session: {session_id}")
    
    # 創建視覺化器
    viz = ExperimentVisualizer(session_id=session_id)
    
    if not viz.entries:
        print("❌ Session 為空")
        return
    
    print(f"📊 該 session 有 {len(viz.entries)} 個查詢記錄")
    
    # 1. 破產機率分布圖
    print(f"\n1. 📊 破產機率分布圖")
    try:
        viz.plot_bankruptcy_distribution(save_path="logs/demo_bankruptcy_dist.png")
        print(f"   ✅ 已保存: logs/demo_bankruptcy_dist.png")
    except Exception as e:
        print(f"   ❌ 生成失敗: {e}")
    
    # 2. 參數影響分析圖
    print(f"\n2. 🔍 參數影響分析圖")
    try:
        viz.plot_parameter_analysis(save_path="logs/demo_param_analysis.png")
        print(f"   ✅ 已保存: logs/demo_param_analysis.png")
    except Exception as e:
        print(f"   ❌ 生成失敗: {e}")
    
    # 3. 蒙特卡洛路徑圖
    print(f"\n3. 📈 蒙特卡洛路徑圖")
    for entry in viz.entries:
        entry_id = entry['id']
        try:
            viz.plot_monte_carlo_paths(
                entry_id=entry_id,
                save_path=f"logs/demo_monte_carlo_{entry_id}.png",
                n_paths=30  # 只顯示 30 條路徑
            )
            print(f"   ✅ 查詢 {entry_id}: logs/demo_monte_carlo_{entry_id}.png")
        except Exception as e:
            print(f"   ❌ 查詢 {entry_id} 失敗: {e}")
    
    # 4. 完整報告
    print(f"\n4. 📋 完整視覺化報告")
    try:
        report_dir = viz.create_session_report(output_dir="logs/demo_full_report")
        print(f"   ✅ 完整報告: {report_dir}")
        
        # 列出報告內容
        if Path(report_dir).exists():
            files = list(Path(report_dir).glob("*"))
            print(f"   📁 包含 {len(files)} 個檔案:")
            for file in sorted(files):
                print(f"      - {file.name}")
    except Exception as e:
        print(f"   ❌ 生成完整報告失敗: {e}")


def demo_usage_examples():
    """使用範例"""
    print(f"\n\n=== ExperimentVisualizer 使用範例 ===\n")
    
    print("1. 基本使用 - 分析當前 session:")
    print("```python")
    print("from utils.visualizer import ExperimentVisualizer")
    print("")
    print("# 使用當前 session")
    print("viz = ExperimentVisualizer()")
    print("viz.create_session_report()")
    print("```")
    
    print("\n2. 分析特定 session:")
    print("```python")
    print("# 使用特定 session ID")
    print("viz = ExperimentVisualizer(session_id='20250630_181812')")
    print("viz.plot_bankruptcy_distribution('my_chart.png')")
    print("```")
    
    print("\n3. 生成個別圖表:")
    print("```python")
    print("viz = ExperimentVisualizer()")
    print("")
    print("# 破產機率分布")
    print("viz.plot_bankruptcy_distribution('bankruptcy.png')")
    print("")
    print("# 參數影響分析")
    print("viz.plot_parameter_analysis('parameters.png')")
    print("")
    print("# 蒙特卡洛路徑 (查詢 1, 顯示 50 條路徑)")
    print("viz.plot_monte_carlo_paths(entry_id=1, n_paths=50, save_path='paths.png')")
    print("```")
    
    print("\n4. 命令列使用:")
    print("```bash")
    print("python -m utils.visualizer --session-id 20250630_181812 --output-dir my_report")
    print("```")
    
    print("\n5. 生成的檔案類型:")
    print("  📊 bankruptcy_distribution.png - 破產機率分布圖")
    print("  🔍 parameter_analysis.png - 參數對破產機率的影響")
    print("  📈 monte_carlo_paths_query_X.png - 個別查詢的蒙特卡洛路徑")
    print("  📈 summary_stats.png - Session 統計圖表")
    print("  📋 summary.md - Markdown 格式的文字報告")


if __name__ == "__main__":
    demo_all_sessions()
    demo_individual_visualizations()
    demo_usage_examples()