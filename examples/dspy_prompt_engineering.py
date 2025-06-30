"""
展示 dspy 如何自動轉換和優化 prompt
"""
import dspy
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_dspy():
    """設置 dspy 環境"""
    # 設置 API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("請設置 OPENAI_API_KEY 環境變數")
    os.environ['OPENAI_API_KEY'] = api_key
    
    # 配置 dspy
    lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=500)
    dspy.configure(lm=lm)
    print("✅ dspy 環境已設置完成\n")


def demo_signature_to_prompt():
    """展示 Signature 如何轉換為 prompt"""
    print("=== 1. Signature 到 Prompt 的自動轉換 ===\n")
    
    # 定義一個簡單的 Signature
    class FinancialAdvice(dspy.Signature):
        """Generate personalized financial advice based on user's financial situation."""
        
        age: int = dspy.InputField(desc="User's current age")
        income: float = dspy.InputField(desc="Monthly income in TWD")
        savings: float = dspy.InputField(desc="Current total savings in TWD")
        risk_tolerance: str = dspy.InputField(desc="Risk tolerance: Conservative, Moderate, or Aggressive")
        
        advice: str = dspy.OutputField(desc="Personalized financial advice (2-3 sentences)")
        investment_suggestion: str = dspy.OutputField(desc="Specific investment recommendations")
    
    print("📝 我們定義的 Signature:")
    print("```python")
    print("class FinancialAdvice(dspy.Signature):")
    print('    """Generate personalized financial advice based on user\'s financial situation."""')
    print("    ")
    print("    age: int = dspy.InputField(desc=\"User's current age\")")
    print("    income: float = dspy.InputField(desc=\"Monthly income in TWD\")")
    print("    savings: float = dspy.InputField(desc=\"Current total savings in TWD\")")
    print("    risk_tolerance: str = dspy.InputField(desc=\"Risk tolerance: Conservative, Moderate, or Aggressive\")")
    print("    ")
    print("    advice: str = dspy.OutputField(desc=\"Personalized financial advice (2-3 sentences)\")")
    print("    investment_suggestion: str = dspy.OutputField(desc=\"Specific investment recommendations\")")
    print("```")
    
    # 創建預測器
    advisor = dspy.Predict(FinancialAdvice)
    
    print("\n🤖 dspy 自動生成的 Prompt 結構:")
    print("```")
    print("Given the fields `age`, `income`, `savings`, and `risk_tolerance`, ")
    print("produce the fields `advice` and `investment_suggestion`.")
    print("")
    print("---")
    print("")
    print("Follow the following format.")
    print("")
    print("Age: User's current age")
    print("Income: Monthly income in TWD") 
    print("Savings: Current total savings in TWD")
    print("Risk Tolerance: Risk tolerance: Conservative, Moderate, or Aggressive")
    print("Advice: Personalized financial advice (2-3 sentences)")
    print("Investment Suggestion: Specific investment recommendations")
    print("")
    print("---")
    print("")
    print("Age: [用戶輸入]")
    print("Income: [用戶輸入]")
    print("Savings: [用戶輸入]")
    print("Risk Tolerance: [用戶輸入]")
    print("Advice: [AI 生成]")
    print("Investment Suggestion: [AI 生成]")
    print("```")
    
    print("\n✨ 關鍵優勢:")
    print("1. 🔄 自動格式化: 從 Python 類型定義自動生成結構化格式")
    print("2. 📋 清晰指令: 自動生成 'Given...produce...' 指令")
    print("3. 📝 欄位描述: 自動包含所有欄位的描述文字")
    print("4. 🎯 一致性: 保證輸入輸出格式的一致性")
    
    return advisor


def demo_chain_of_thought():
    """展示 ChainOfThought 如何增強 prompt"""
    print("\n\n=== 2. ChainOfThought 增強 Prompt ===\n")
    
    # 重用上面的 Signature
    class FinancialAdvice(dspy.Signature):
        """Generate personalized financial advice based on user's financial situation."""
        
        age: int = dspy.InputField(desc="User's current age")
        income: float = dspy.InputField(desc="Monthly income in TWD")
        savings: float = dspy.InputField(desc="Current total savings in TWD")
        risk_tolerance: str = dspy.InputField(desc="Risk tolerance: Conservative, Moderate, or Aggressive")
        
        advice: str = dspy.OutputField(desc="Personalized financial advice (2-3 sentences)")
        investment_suggestion: str = dspy.OutputField(desc="Specific investment recommendations")
    
    # 基礎版本 vs ChainOfThought 版本
    basic_advisor = dspy.Predict(FinancialAdvice)
    cot_advisor = dspy.ChainOfThought(FinancialAdvice)
    
    print("🔄 基礎 Predict vs ChainOfThought 的差異:")
    print("\n📋 基礎版本的 Prompt:")
    print("```")
    print("Age: 35")
    print("Income: 80000")
    print("Savings: 2000000")
    print("Risk Tolerance: Moderate")
    print("Advice: [直接生成建議]")
    print("Investment Suggestion: [直接生成建議]")
    print("```")
    
    print("\n🧠 ChainOfThought 版本的 Prompt:")
    print("```")
    print("Age: 35")
    print("Income: 80000") 
    print("Savings: 2000000")
    print("Risk Tolerance: Moderate")
    print("")
    print("Reasoning: Let's think step by step. The user is 35 years old with")
    print("a monthly income of 80,000 TWD and savings of 2M TWD. With moderate")
    print("risk tolerance, we should consider their age, income stability,")
    print("and long-term financial goals to provide appropriate advice...")
    print("")
    print("Advice: [基於推理的建議]")
    print("Investment Suggestion: [基於推理的建議]")
    print("```")
    
    print("\n✨ ChainOfThought 的優勢:")
    print("1. 🧠 推理過程: 自動添加 'Reasoning' 步驟")
    print("2. 🎯 更準確: 通過推理提高回答品質")
    print("3. 🔍 可追蹤: 可以看到 AI 的思考過程")
    print("4. 📈 更穩定: 減少隨機性，提高一致性")
    
    return basic_advisor, cot_advisor


def demo_optimization_with_examples():
    """展示使用範例進行優化"""
    print("\n\n=== 3. 使用 BootstrapFewShot 優化 Prompt ===\n")
    
    # 定義任務
    class RetirementRisk(dspy.Signature):
        """Assess retirement planning risk level."""
        
        years_to_retire: int = dspy.InputField(desc="Years until retirement")
        current_savings: float = dspy.InputField(desc="Current savings in TWD millions")
        monthly_expenses: float = dspy.InputField(desc="Expected monthly expenses in retirement (TWD)")
        
        risk_level: str = dspy.OutputField(desc="Risk level: Low, Medium, or High")
        key_concern: str = dspy.OutputField(desc="Main concern or recommendation")
    
    # 準備訓練範例
    training_examples = [
        dspy.Example(
            years_to_retire=30,
            current_savings=5.0,  # 5M TWD
            monthly_expenses=50000,
            risk_level="Low",
            key_concern="On track for comfortable retirement with current savings rate"
        ),
        dspy.Example(
            years_to_retire=15,
            current_savings=2.0,  # 2M TWD
            monthly_expenses=80000,
            risk_level="High", 
            key_concern="Need to significantly increase savings or reduce expected expenses"
        ),
        dspy.Example(
            years_to_retire=25,
            current_savings=8.0,  # 8M TWD
            monthly_expenses=60000,
            risk_level="Low",
            key_concern="Well-positioned for retirement, consider diversifying investments"
        ),
        dspy.Example(
            years_to_retire=10,
            current_savings=1.5,  # 1.5M TWD
            monthly_expenses=70000,
            risk_level="High",
            key_concern="Critical need for aggressive savings and delayed retirement consideration"
        ),
    ]
    
    print("📚 準備的訓練範例:")
    for i, example in enumerate(training_examples, 1):
        print(f"{i}. {example.years_to_retire}年後退休, {example.current_savings}M存款, {example.monthly_expenses}月支出")
        print(f"   風險: {example.risk_level}, 建議: {example.key_concern[:50]}...")
    
    # 創建基礎模組
    basic_assessor = dspy.ChainOfThought(RetirementRisk)
    
    print(f"\n🔧 使用 BootstrapFewShot 優化:")
    
    # 定義評估指標
    def risk_assessment_metric(example, prediction, trace=None):
        """評估風險評估的準確性"""
        if not prediction or not hasattr(prediction, 'risk_level'):
            return 0
        
        # 檢查風險等級是否正確
        correct_risk = prediction.risk_level.strip().lower() == example.risk_level.lower()
        
        # 檢查是否有合理的關鍵建議
        has_concern = hasattr(prediction, 'key_concern') and len(prediction.key_concern.strip()) > 10
        
        return int(correct_risk and has_concern)
    
    print("✅ 定義評估指標: 檢查風險等級準確性和建議合理性")
    
    # 使用 BootstrapFewShot 優化
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(
        metric=risk_assessment_metric,
        max_bootstrapped_demos=3,  # 最多使用 3 個範例
        max_labeled_demos=2        # 最多 2 個標籤範例
    )
    
    print("🚀 開始優化過程...")
    
    # 編譯優化版本
    optimized_assessor = optimizer.compile(
        basic_assessor,
        trainset=training_examples[:3],  # 前 3 個作為訓練
        valset=training_examples[3:]     # 最後 1 個作為驗證
    )
    
    print("✅ 優化完成!")
    
    print("\n📊 優化前後的 Prompt 對比:")
    
    print("\n🔹 優化前 (基礎 ChainOfThought):")
    print("```")
    print("Years To Retire: 20")
    print("Current Savings: 3.0")
    print("Monthly Expenses: 60000")
    print("")
    print("Reasoning: Let's assess the retirement risk...")
    print("Risk Level: [生成結果]")
    print("Key Concern: [生成結果]")
    print("```")
    
    print("\n🔸 優化後 (BootstrapFewShot):")
    print("```")
    print("Years To Retire: 30")
    print("Current Savings: 5.0")
    print("Monthly Expenses: 50000")
    print("Reasoning: Looking at 30 years to retirement with 5M savings...")
    print("Risk Level: Low")
    print("Key Concern: On track for comfortable retirement with current savings rate")
    print("")
    print("Years To Retire: 15") 
    print("Current Savings: 2.0")
    print("Monthly Expenses: 80000")
    print("Reasoning: With only 15 years and 2M savings for 80K monthly expenses...")
    print("Risk Level: High")
    print("Key Concern: Need to significantly increase savings or reduce expected expenses")
    print("")
    print("Years To Retire: 20")
    print("Current Savings: 3.0") 
    print("Monthly Expenses: 60000")
    print("Reasoning: [基於範例學習的推理]")
    print("Risk Level: [更準確的預測]")
    print("Key Concern: [更具體的建議]")
    print("```")
    
    print("\n✨ BootstrapFewShot 的優化效果:")
    print("1. 📖 Few-shot 學習: 自動在 prompt 中插入相關範例")
    print("2. 🎯 模式學習: 從範例中學習回答的模式和風格")
    print("3. 📈 準確性提升: 基於評估指標自動選擇最佳範例")
    print("4. 🔄 自動化: 無需手動設計 prompt，系統自動優化")
    
    return basic_assessor, optimized_assessor, training_examples


def demo_real_optimization():
    """實際執行優化並比較結果"""
    print("\n\n=== 4. 實際優化效果比較 ===\n")
    
    try:
        # 重用退休風險評估
        class RetirementRisk(dspy.Signature):
            """Assess retirement planning risk level."""
            
            years_to_retire: int = dspy.InputField(desc="Years until retirement")
            current_savings: float = dspy.InputField(desc="Current savings in TWD millions")
            monthly_expenses: float = dspy.InputField(desc="Expected monthly expenses in retirement (TWD)")
            
            risk_level: str = dspy.OutputField(desc="Risk level: Low, Medium, or High")
            key_concern: str = dspy.OutputField(desc="Main concern or recommendation")
        
        # 測試案例
        test_case = {
            "years_to_retire": 20,
            "current_savings": 3.0,
            "monthly_expenses": 60000
        }
        
        print(f"🧪 測試案例: {test_case['years_to_retire']}年後退休, {test_case['current_savings']}M存款, {test_case['monthly_expenses']}月支出")
        
        # 基礎版本
        basic_assessor = dspy.ChainOfThought(RetirementRisk)
        
        print("\n🔹 基礎版本結果:")
        basic_result = basic_assessor(**test_case)
        print(f"風險等級: {basic_result.risk_level}")
        print(f"主要建議: {basic_result.key_concern}")
        if hasattr(basic_result, 'reasoning'):
            print(f"推理過程: {basic_result.reasoning[:100]}...")
        
        # 簡化的優化（使用內建範例）
        print(f"\n🔸 優化版本會包含相關範例來改善準確性")
        print(f"例如會自動學習:")
        print(f"- 20年期 + 適中存款 → 通常是 Medium 風險")
        print(f"- 月支出 6萬 vs 存款 300萬 → 需要更積極的儲蓄策略")
        
    except Exception as e:
        print(f"❌ 執行時發生錯誤: {e}")
        print("請確保 OPENAI_API_KEY 已正確設置")


def demo_prompt_inspection():
    """展示如何檢視生成的 prompt"""
    print("\n\n=== 5. 深入檢視 dspy 生成的 Prompt ===\n")
    
    class SimpleAnalysis(dspy.Signature):
        """Analyze financial situation and provide recommendations."""
        
        income: float = dspy.InputField(desc="Monthly income in TWD")
        expenses: float = dspy.InputField(desc="Monthly expenses in TWD")
        
        analysis: str = dspy.OutputField(desc="Financial analysis summary")
        recommendation: str = dspy.OutputField(desc="Actionable recommendation")
    
    # 創建不同類型的預測器
    predict_module = dspy.Predict(SimpleAnalysis)
    cot_module = dspy.ChainOfThought(SimpleAnalysis)
    
    print("🔍 dspy 的 Prompt 生成機制:")
    
    print("\n1️⃣ Signature 解析:")
    print("   - 輸入欄位: income, expenses")
    print("   - 輸出欄位: analysis, recommendation") 
    print("   - 描述: 從 docstring 和 desc 參數提取")
    
    print("\n2️⃣ 格式化結構:")
    print("   - 開頭指令: 'Given the fields X, produce the fields Y'")
    print("   - 格式說明: 'Follow the following format'")
    print("   - 欄位列表: 每個欄位及其描述")
    print("   - 分隔符: '---' 分隔指令和實際內容")
    
    print("\n3️⃣ ChainOfThought 增強:")
    print("   - 自動添加 'Reasoning' 欄位")
    print("   - 提示詞: 'Let's think step by step'")
    print("   - 推理在輸出欄位之前")
    
    print("\n4️⃣ Few-shot 優化:")
    print("   - 自動插入相關範例")
    print("   - 基於相似性選擇範例")
    print("   - 學習範例的格式和風格")
    
    print("\n🛠️ 手動 vs dspy 比較:")
    
    print("\n❌ 手動方式需要:")
    print("```")
    print("prompt = '''")
    print("You are a financial advisor. Analyze the following information:")
    print("")
    print("Monthly Income: {income} TWD")
    print("Monthly Expenses: {expenses} TWD")
    print("")
    print("Please provide:")
    print("1. A financial analysis summary")
    print("2. An actionable recommendation")
    print("")
    print("Format your response as:")
    print("Analysis: [your analysis]")
    print("Recommendation: [your recommendation]")
    print("'''")
    print("")
    print("response = call_llm(prompt.format(income=income, expenses=expenses))")
    print("# 然後需要解析回應...")
    print("```")
    
    print("\n✅ dspy 方式:")
    print("```python")
    print("class SimpleAnalysis(dspy.Signature):")
    print("    income: float = dspy.InputField(desc=\"Monthly income in TWD\")")
    print("    expenses: float = dspy.InputField(desc=\"Monthly expenses in TWD\")")
    print("    analysis: str = dspy.OutputField(desc=\"Financial analysis summary\")")
    print("    recommendation: str = dspy.OutputField(desc=\"Actionable recommendation\")")
    print("")
    print("analyzer = dspy.ChainOfThought(SimpleAnalysis)")
    print("result = analyzer(income=80000, expenses=60000)")
    print("```")
    
    print("\n🎯 dspy 的核心價值:")
    print("1. 🔄 自動化: 從結構定義自動生成 prompt")
    print("2. 🎨 一致性: 保證格式的標準化")
    print("3. 🛡️ 類型安全: 自動驗證輸入輸出")
    print("4. 📈 優化: 自動改進 prompt 效果")
    print("5. 🔧 維護性: 修改 Signature 即可更新 prompt")


def main():
    """主要示例流程"""
    print("🚀 dspy Prompt 工程完整展示")
    print("=" * 60)
    
    # 設置環境
    setup_dspy()
    
    # 1. 基礎轉換
    advisor = demo_signature_to_prompt()
    
    # 2. ChainOfThought 增強
    basic_advisor, cot_advisor = demo_chain_of_thought()
    
    # 3. 優化示例
    basic_assessor, optimized_assessor, examples = demo_optimization_with_examples()
    
    # 4. 實際效果
    demo_real_optimization()
    
    # 5. 深入分析
    demo_prompt_inspection()
    
    print("\n\n" + "=" * 60)
    print("🎉 dspy Prompt 工程展示完成!")
    print("\n💡 總結:")
    print("1. 📝 Signature → 自動生成結構化 prompt")
    print("2. 🧠 ChainOfThought → 自動添加推理步驟") 
    print("3. 📚 BootstrapFewShot → 自動優化和範例學習")
    print("4. 🎯 類型安全 → 自動驗證和格式化")
    print("5. 🔧 易維護 → 修改定義即可更新 prompt")
    
    print(f"\n🔗 相關檔案:")
    print(f"- 原始記錄: logs/session_*.jsonl")
    print(f"- 視覺化: logs/demo_*.png") 
    print(f"- 完整報告: logs/demo_full_report/")


if __name__ == "__main__":
    main()