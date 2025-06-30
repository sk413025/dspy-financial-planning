#!/usr/bin/env python3
"""
日誌查看工具 - 以易讀格式顯示 dspy 實驗日誌
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def format_timestamp(timestamp_str):
    """格式化時間戳為易讀格式"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def print_section(title, content="", level=1):
    """打印章節標題"""
    if level == 1:
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print(f"{'='*60}")
    elif level == 2:
        print(f"\n{'-'*40}")
        print(f"🔸 {title}")
        print(f"{'-'*40}")
    else:
        print(f"\n• {title}")
    
    if content:
        print(content)


def display_log_entry(entry):
    """顯示單個日誌條目"""
    
    # 基本資訊
    print_section("基本資訊")
    print(f"ID: {entry['id']}")
    print(f"時間: {format_timestamp(entry['timestamp'])}")
    print(f"來源: {entry['source']}")
    print(f"查詢: {entry['input']['raw_query']}")
    
    if entry.get('duration_ms'):
        print(f"執行時間: {entry['duration_ms']:.0f} ms")
    
    # 步驟資訊
    if 'steps' in entry.get('intermediate', {}):
        print_section("執行步驟")
        for step in entry['intermediate']['steps']:
            print_section(f"步驟 {step['step']}: {step['title']}", level=2)
            
            if 'signature_definition' in step:
                sig = step['signature_definition']
                print(f"📝 Signature 類別: {sig['class']}")
                print(f"📄 描述: {sig['docstring']}")
                
                print("\n🔹 輸入欄位:")
                for field in sig['input_fields']:
                    print(f"  • {field['name']} ({field['type']}): {field['desc']}")
                
                print("\n🔹 輸出欄位:")
                for field in sig['output_fields']:
                    print(f"  • {field['name']} ({field['type']}): {field['desc']}")
            
            if 'generated_prompt' in step:
                print(f"\n🤖 自動生成的 Prompt:")
                print("```")
                print(step['generated_prompt'])
                print("```")
    
    # 測試參數
    if 'parameters' in entry.get('intermediate', {}):
        print_section("測試參數")
        for method, param_data in entry['intermediate']['parameters'].items():
            print_section(f"{method.replace('_', ' ').title()}", level=2)
            params = param_data['params']
            for key, value in params.items():
                if isinstance(value, float):
                    print(f"  • {key}: {value:,.1f}")
                elif isinstance(value, int):
                    print(f"  • {key}: {value:,}")
                else:
                    print(f"  • {key}: {value}")
    
    # 預測結果
    if 'predictions' in entry.get('intermediate', {}):
        print_section("預測結果")
        for pred in entry['intermediate']['predictions']:
            method = pred['method'].replace('_', ' ').title()
            print_section(f"{method} 結果", level=2)
            print(f"🎯 風險等級: {pred['risk_level']}")
            print(f"💰 建議儲蓄: {pred['monthly_save_needed']}")
            print(f"📋 策略: {pred['strategy']}")
            
            if pred.get('has_reasoning') and pred.get('reasoning'):
                print(f"\n🧠 推理過程:")
                print(f"{pred['reasoning']}")
    
    # 訓練範例
    if 'training_examples' in entry.get('intermediate', {}):
        examples = entry['intermediate']['training_examples']
        print_section(f"訓練範例 (共 {examples['count']} 個)")
        for i, ex in enumerate(examples['examples'], 1):
            print(f"\n{i}. 年齡 {ex['age']} 歲, {ex['savings']} 萬存款, 月收入 {ex['monthly_income']:,} TWD")
            print(f"   風險: {ex['risk_level']}")
            print(f"   建議: {ex['monthly_save_needed']}")
            print(f"   策略: {ex['strategy']}")
    
    # 結果比較
    if 'comparison' in entry.get('intermediate', {}):
        comp = entry['intermediate']['comparison']
        print_section("結果比較")
        
        methods = ['basic_predict', 'chain_of_thought', 'few_shot_optimized']
        method_names = ['基礎版本', 'ChainOfThought', 'Few-Shot 優化']
        
        # 比較表格
        print("\n📊 風險等級比較:")
        for method, name in zip(methods, method_names):
            if method in comp:
                risk = comp[method].get('risk_level', '無資料')
                print(f"  • {name}: {risk}")
        
        print("\n💰 儲蓄建議比較:")
        for method, name in zip(methods, method_names):
            if method in comp:
                save = comp[method].get('monthly_save_needed', '無資料')
                print(f"  • {name}: {save}")
    
    # 演示總結
    if 'demo_summary' in entry.get('output', {}):
        summary = entry['output']['demo_summary']
        print_section("演示總結")
        print(f"✅ 演示完成: {summary['demo_completed']}")
        print(f"✅ 基礎版本成功: {summary['basic_predict_success']}")
        print(f"✅ ChainOfThought 成功: {summary['chain_of_thought_success']}")
        print(f"✅ Few-Shot 優化成功: {summary['few_shot_success']}")
        
        print("\n💡 關鍵洞察:")
        for insight in summary['key_insights']:
            print(f"  • {insight}")
        
        print("\n🚀 建議:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    # 錯誤資訊
    if entry.get('errors'):
        print_section("錯誤記錄")
        for error in entry['errors']:
            print(f"❌ [{error['phase']}] {error['error']}")
            print(f"   時間: {format_timestamp(error['timestamp'])}")


def main():
    if len(sys.argv) < 2:
        # 列出可用的日誌檔案
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("❌ logs 目錄不存在")
            return
        
        jsonl_files = list(logs_dir.glob("session_*.jsonl"))
        if not jsonl_files:
            print("❌ 未找到日誌檔案")
            return
        
        print("📁 可用的日誌檔案:")
        for i, file in enumerate(sorted(jsonl_files, reverse=True), 1):
            print(f"  {i}. {file.name}")
        
        print(f"\n使用方式: python {sys.argv[0]} <檔案名稱>")
        print(f"例如: python {sys.argv[0]} {jsonl_files[0].name}")
        return
    
    # 讀取指定的日誌檔案
    log_file = Path("logs") / sys.argv[1]
    if not log_file.exists():
        print(f"❌ 檔案不存在: {log_file}")
        return
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    entry = json.loads(line)
                    print(f"\n🔍 日誌條目 {line_num}")
                    print("=" * 80)
                    display_log_entry(entry)
                    print("\n" + "=" * 80)
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析錯誤: {e}")
    except Exception as e:
        print(f"❌ 讀取錯誤: {e}")


if __name__ == "__main__":
    main()