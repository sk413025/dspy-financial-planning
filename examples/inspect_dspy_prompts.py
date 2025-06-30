"""
實際檢視 dspy 內部生成的 prompt
"""
import dspy
import os

def setup_dspy():
    """設置 dspy"""
    try:
        import sys
        sys.path.append('..')
        from config import OPENAI_API_KEY
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    except ImportError:
        raise ValueError("請複製 config.example.py 為 config.py 並設置您的 API key")
    
    lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=300)
    dspy.configure(lm=lm)

def inspect_signature_conversion():
    """檢視 Signature 如何轉換為 prompt"""
    print("=== 🔍 實際檢視 dspy 生成的 Prompt ===\n")
    
    # 定義我們的退休規劃 Signature
    class RetirementPlanning(dspy.Signature):
        """Analyze retirement planning scenario and provide recommendations."""
        
        age: int = dspy.InputField(desc="Current age")
        years_to_retire: int = dspy.InputField(desc="Years until planned retirement")
        savings: float = dspy.InputField(desc="Current savings in TWD")
        monthly_income: float = dspy.InputField(desc="Current monthly income in TWD")
        
        risk_level: str = dspy.OutputField(desc="Risk assessment: Low, Medium, or High")
        monthly_savings_needed: str = dspy.OutputField(desc="Recommended monthly savings amount")
        strategy: str = dspy.OutputField(desc="Retirement strategy recommendation")
    
    print("📝 我們定義的 Signature:")
    print("```python")
    print("class RetirementPlanning(dspy.Signature):")
    print('    """Analyze retirement planning scenario and provide recommendations."""')
    print("    age: int = dspy.InputField(desc=\"Current age\")")
    print("    years_to_retire: int = dspy.InputField(desc=\"Years until planned retirement\")")
    print("    savings: float = dspy.InputField(desc=\"Current savings in TWD\")")
    print("    monthly_income: float = dspy.InputField(desc=\"Current monthly income in TWD\")")
    print("    ")
    print("    risk_level: str = dspy.OutputField(desc=\"Risk assessment: Low, Medium, or High\")")
    print("    monthly_savings_needed: str = dspy.OutputField(desc=\"Recommended monthly savings amount\")")
    print("    strategy: str = dspy.OutputField(desc=\"Retirement strategy recommendation\")")
    print("```")
    
    return RetirementPlanning

def demonstrate_prompt_generation():
    """展示實際的 prompt 生成過程"""
    print("\n\n=== 📋 Step-by-Step Prompt 生成過程 ===\n")
    
    class SimpleRetirement(dspy.Signature):
        """Provide retirement advice."""
        age: int = dspy.InputField(desc="Age in years")
        savings: float = dspy.InputField(desc="Savings in TWD") 
        advice: str = dspy.OutputField(desc="Retirement advice")
    
    print("🔄 dspy 的 Prompt 生成步驟:")
    
    print("\n1️⃣ 解析 Signature:")
    print("   - 類別名稱: SimpleRetirement")
    print("   - Docstring: 'Provide retirement advice.'")
    print("   - 輸入欄位: age (int), savings (float)")
    print("   - 輸出欄位: advice (str)")
    
    print("\n2️⃣ 生成基礎指令:")
    print("   - 'Given the fields `age` and `savings`, produce the field `advice`.'")
    
    print("\n3️⃣ 建立格式說明:")
    print("   - 'Follow the following format.'")
    print("   - 列出所有欄位及其描述")
    
    print("\n4️⃣ 完整生成的 Prompt:")
    print("```")
    print("Provide retirement advice.")
    print("")
    print("---")
    print("")
    print("Follow the following format.")
    print("")
    print("Age: Age in years")
    print("Savings: Savings in TWD")
    print("Advice: Retirement advice")
    print("")
    print("---")
    print("")
    print("Age: 45")
    print("Savings: 5000000.0")
    print("Advice:")
    print("```")

def demonstrate_chain_of_thought_prompt():
    """展示 ChainOfThought 如何修改 prompt"""
    print("\n\n=== 🧠 ChainOfThought Prompt 修改 ===\n")
    
    class SimpleRetirement(dspy.Signature):
        """Provide retirement advice."""
        age: int = dspy.InputField(desc="Age in years")
        savings: float = dspy.InputField(desc="Savings in TWD")
        advice: str = dspy.OutputField(desc="Retirement advice")
    
    print("🔄 ChainOfThought 如何修改 Prompt:")
    
    print("\n📋 基礎 Predict 版本:")
    print("```")
    print("Age: 45")
    print("Savings: 5000000.0")
    print("Advice: [直接生成]")
    print("```")
    
    print("\n🧠 ChainOfThought 修改後:")
    print("```") 
    print("Age: 45")
    print("Savings: 5000000.0")
    print("")
    print("Reasoning: Let's think step by step in order to provide retirement advice.")
    print("The person is 45 years old with 5,000,000 TWD in savings...")
    print("")
    print("Advice: [基於推理的建議]")
    print("```")
    
    print("\n✨ 關鍵變化:")
    print("1. ➕ 自動添加 'Reasoning' 欄位")
    print("2. 📝 插入 'Let's think step by step' 提示")
    print("3. 🔗 推理欄位連結到最終輸出")
    print("4. 🎯 引導模型進行結構化思考")

def show_few_shot_enhancement():
    """展示 Few-shot 如何增強 prompt"""
    print("\n\n=== 📚 Few-Shot 學習增強 ===\n")
    
    print("🎯 Few-Shot 優化的 Prompt 變化:")
    
    print("\n📋 優化前 (零樣本):")
    print("```")
    print("Age: 35")
    print("Savings: 2000000.0")
    print("Risk Level:")
    print("```")
    
    print("\n📚 優化後 (Few-Shot):")
    print("```")
    print("Age: 25")
    print("Savings: 500000.0")
    print("Risk Level: High")
    print("")
    print("Age: 55")  
    print("Savings: 10000000.0")
    print("Risk Level: Low")
    print("")
    print("Age: 40")
    print("Savings: 3000000.0") 
    print("Risk Level: Medium")
    print("")
    print("Age: 35")
    print("Savings: 2000000.0")
    print("Risk Level:")
    print("```")
    
    print("\n✨ Few-Shot 的優勢:")
    print("1. 📖 學習模式: 從範例中學習分類邏輯")
    print("2. 🎯 上下文引導: 提供具體的參考標準")
    print("3. 📈 一致性: 確保回答風格的統一")
    print("4. 🎨 品質提升: 顯著改善輸出品質")

def demonstrate_actual_usage():
    """展示實際使用"""
    print("\n\n=== 🚀 實際使用示例 ===\n")
    
    class RetirementAdvice(dspy.Signature):
        """Provide personalized retirement planning advice."""
        
        age: int = dspy.InputField(desc="Current age")
        savings: float = dspy.InputField(desc="Current savings in TWD")
        
        risk: str = dspy.OutputField(desc="Risk level: Low/Medium/High")
        advice: str = dspy.OutputField(desc="Specific advice")
    
    print("🧪 測試實際執行:")
    
    try:
        setup_dspy()
        
        # 創建 ChainOfThought 版本
        advisor = dspy.ChainOfThought(RetirementAdvice)
        
        # 執行測試
        result = advisor(age=42, savings=4500000.0)
        
        print("✅ 執行成功!")
        print(f"輸入: 42歲, 450萬存款")
        print(f"風險評估: {result.risk}")
        print(f"建議: {result.advice}")
        
        if hasattr(result, 'reasoning'):
            print(f"推理過程: {result.reasoning[:200]}...")
        
        print("\n🔍 實際發送給 GPT 的 Prompt (概念):")
        print("```")
        print("Provide personalized retirement planning advice.")
        print("")
        print("---")
        print("")
        print("Follow the following format.")
        print("")
        print("Age: Current age")
        print("Savings: Current savings in TWD")
        print("Reasoning: Let's think step by step in order to provide personalized retirement planning advice.")
        print("Risk: Risk level: Low/Medium/High")
        print("Advice: Specific advice")
        print("")
        print("---")
        print("")
        print("Age: 42")
        print("Savings: 4500000.0")
        print("Reasoning:")
        print("```")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")

def show_optimization_workflow():
    """展示優化工作流程"""
    print("\n\n=== 🔧 優化工作流程 ===\n")
    
    print("🔄 完整的 dspy 優化流程:")
    
    print("\n1️⃣ 定義 Signature:")
    print("```python")
    print("class TaskSignature(dspy.Signature):")
    print("    input_field: type = dspy.InputField(desc='...')")
    print("    output_field: type = dspy.OutputField(desc='...')")
    print("```")
    
    print("\n2️⃣ 選擇預測器類型:")
    print("```python")
    print("# 基礎版本")
    print("predictor = dspy.Predict(TaskSignature)")
    print("")
    print("# 推理增強版本")
    print("predictor = dspy.ChainOfThought(TaskSignature)")
    print("```")
    
    print("\n3️⃣ 準備訓練資料:")
    print("```python")
    print("examples = [")
    print("    dspy.Example(input_field=..., output_field=...),")
    print("    # 更多範例...")
    print("]")
    print("```")
    
    print("\n4️⃣ 設定優化器:")
    print("```python")
    print("from dspy.teleprompt import BootstrapFewShot")
    print("")
    print("optimizer = BootstrapFewShot(metric=evaluation_function)")
    print("optimized_predictor = optimizer.compile(predictor, trainset=examples)")
    print("```")
    
    print("\n5️⃣ 使用優化後的模組:")
    print("```python")
    print("result = optimized_predictor(input_field=value)")
    print("print(result.output_field)")
    print("```")
    
    print("\n✨ 整個過程的價值:")
    print("1. 🔄 自動化: 從定義到優化全自動")
    print("2. 📈 漸進式: 從簡單到複雜逐步改進")
    print("3. 🎯 目標導向: 基於評估指標優化")
    print("4. 🔧 可維護: 易於修改和擴展")

def main():
    """主程式"""
    print("🔍 dspy Prompt 內部機制深度剖析")
    print("=" * 60)
    
    # 展示各個方面
    signature = inspect_signature_conversion()
    demonstrate_prompt_generation()
    demonstrate_chain_of_thought_prompt()
    show_few_shot_enhancement()
    demonstrate_actual_usage()
    show_optimization_workflow()
    
    print("\n" + "=" * 60)
    print("🎉 dspy Prompt 機制剖析完成!")
    
    print("\n🎯 關鍵洞察:")
    print("1. 📝 Signature 是 dspy 的核心 - 定義輸入輸出結構")
    print("2. 🤖 Prompt 完全自動生成 - 無需手動撰寫")
    print("3. 🧠 ChainOfThought 自動添加推理 - 提高準確性")
    print("4. 📚 Few-Shot 自動選擇範例 - 基於相似性學習")
    print("5. 🔧 優化是系統性的 - 不是隨機調整")
    
    print("\n🚀 實用建議:")
    print("1. 📋 清晰定義 Signature - 好的定義是成功的一半")
    print("2. 🧪 從簡單開始 - 先用 Predict，再用 ChainOfThought")
    print("3. 📊 收集好資料 - 品質資料是優化的基礎")
    print("4. 🎯 定義好指標 - 明確的評估標準很重要")
    print("5. 🔄 持續迭代 - 不斷改進和優化")
    
    print(f"\n💡 記住: dspy 讓你專注於 '做什麼'，而不是 '怎麼做'！")

if __name__ == "__main__":
    main()