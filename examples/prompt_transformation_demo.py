"""
dspy Prompt 轉換和優化的實際展示
"""
import dspy
import os

def setup_dspy():
    """設置 dspy"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("請設置 OPENAI_API_KEY 環境變數")
    os.environ['OPENAI_API_KEY'] = api_key
    
    lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=300)
    dspy.configure(lm=lm)
    print("✅ dspy 已配置完成\n")

def demo_signature_transformation():
    """展示 Signature 到 Prompt 的轉換"""
    print("=== 📝 Signature 轉換為 Prompt ===\n")
    
    # 定義 Signature
    class RetirementAdvice(dspy.Signature):
        """Provide retirement planning advice based on financial parameters."""
        
        current_age: int = dspy.InputField(desc="Current age of the person")
        target_retirement_age: int = dspy.InputField(desc="Desired retirement age")
        current_savings: float = dspy.InputField(desc="Current total savings in TWD")
        monthly_income: float = dspy.InputField(desc="Current monthly income in TWD")
        expected_expenses: float = dspy.InputField(desc="Expected monthly expenses in retirement in TWD")
        
        risk_assessment: str = dspy.OutputField(desc="Risk level: Low, Medium, High")
        savings_recommendation: str = dspy.OutputField(desc="Monthly savings recommendation")
        strategy: str = dspy.OutputField(desc="Overall retirement strategy")
    
    print("📋 我們定義的 Signature:")
    print("```python")
    print("class RetirementAdvice(dspy.Signature):")
    print('    """Provide retirement planning advice based on financial parameters."""')
    print("    current_age: int = dspy.InputField(desc=\"Current age of the person\")")
    print("    target_retirement_age: int = dspy.InputField(desc=\"Desired retirement age\")")
    print("    current_savings: float = dspy.InputField(desc=\"Current total savings in TWD\")")
    print("    # ... 更多欄位")
    print("```")
    
    print("\n🤖 dspy 自動生成的完整 Prompt:")
    print("```")
    print("Provide retirement planning advice based on financial parameters.")
    print("")
    print("---")
    print("")
    print("Follow the following format.")
    print("")
    print("Current Age: Current age of the person")
    print("Target Retirement Age: Desired retirement age")
    print("Current Savings: Current total savings in TWD")
    print("Monthly Income: Current monthly income in TWD")
    print("Expected Expenses: Expected monthly expenses in retirement in TWD")
    print("Risk Assessment: Risk level: Low, Medium, High")
    print("Savings Recommendation: Monthly savings recommendation")
    print("Strategy: Overall retirement strategy")
    print("")
    print("---")
    print("")
    print("Current Age: 35")
    print("Target Retirement Age: 65")
    print("Current Savings: 2000000.0")
    print("Monthly Income: 80000.0")
    print("Expected Expenses: 50000.0")
    print("Risk Assessment: [AI 生成]")
    print("Savings Recommendation: [AI 生成]")
    print("Strategy: [AI 生成]")
    print("```")
    
    return RetirementAdvice

def demo_chain_of_thought_enhancement():
    """展示 ChainOfThought 如何增強 Prompt"""
    print("\n\n=== 🧠 ChainOfThought 增強 ===\n")
    
    class RetirementAdvice(dspy.Signature):
        """Provide retirement planning advice based on financial parameters."""
        
        current_age: int = dspy.InputField(desc="Current age of the person")
        target_retirement_age: int = dspy.InputField(desc="Desired retirement age")
        current_savings: float = dspy.InputField(desc="Current total savings in TWD")
        monthly_income: float = dspy.InputField(desc="Current monthly income in TWD")
        expected_expenses: float = dspy.InputField(desc="Expected monthly expenses in retirement in TWD")
        
        risk_assessment: str = dspy.OutputField(desc="Risk level: Low, Medium, High")
        savings_recommendation: str = dspy.OutputField(desc="Monthly savings recommendation")
        strategy: str = dspy.OutputField(desc="Overall retirement strategy")
    
    # 基礎版本
    basic_advisor = dspy.Predict(RetirementAdvice)
    
    # ChainOfThought 版本
    cot_advisor = dspy.ChainOfThought(RetirementAdvice)
    
    print("🔄 Predict vs ChainOfThought 的 Prompt 差異:")
    
    print("\n📋 基礎 Predict 版本:")
    print("```")
    print("Current Age: 35")
    print("Target Retirement Age: 65") 
    print("Current Savings: 2000000.0")
    print("Monthly Income: 80000.0")
    print("Expected Expenses: 50000.0")
    print("Risk Assessment: Medium")
    print("Savings Recommendation: Save 15,000 TWD monthly")
    print("Strategy: Balanced investment approach")
    print("```")
    
    print("\n🧠 ChainOfThought 增強版本:")
    print("```")
    print("Current Age: 35")
    print("Target Retirement Age: 65")
    print("Current Savings: 2000000.0") 
    print("Monthly Income: 80000.0")
    print("Expected Expenses: 50000.0")
    print("")
    print("Reasoning: Let's analyze this retirement scenario step by step.")
    print("The person is 35 years old and wants to retire at 65, giving them")
    print("30 years to prepare. With current savings of 2M TWD and monthly")
    print("income of 80K TWD, expecting 50K monthly expenses in retirement.")
    print("Need to calculate if current savings trajectory is sufficient...")
    print("")
    print("Risk Assessment: Medium")
    print("Savings Recommendation: Save 15,000 TWD monthly")
    print("Strategy: Balanced investment approach with regular reviews")
    print("```")
    
    print("\n✨ ChainOfThought 的價值:")
    print("1. 🧠 顯示推理過程 - 可以追蹤 AI 的思考邏輯")
    print("2. 🎯 提高準確性 - 逐步分析減少錯誤")
    print("3. 🔍 增加透明度 - 解釋為什麼做出特定建議")
    print("4. 📈 更穩定輸出 - 結構化思考提高一致性")
    
    return basic_advisor, cot_advisor

def demo_actual_execution():
    """實際執行並展示結果"""
    print("\n\n=== 🚀 實際執行效果對比 ===\n")
    
    class RetirementAdvice(dspy.Signature):
        """Provide retirement planning advice based on financial parameters."""
        
        current_age: int = dspy.InputField(desc="Current age of the person")
        current_savings: float = dspy.InputField(desc="Current total savings in TWD")
        monthly_income: float = dspy.InputField(desc="Current monthly income in TWD")
        
        risk_assessment: str = dspy.OutputField(desc="Risk level: Low, Medium, High")
        advice: str = dspy.OutputField(desc="Key retirement planning advice")
    
    # 測試案例
    test_case = {
        "current_age": 40,
        "current_savings": 3000000.0,
        "monthly_income": 100000.0
    }
    
    print(f"🧪 測試案例:")
    print(f"年齡: {test_case['current_age']} 歲")
    print(f"存款: {test_case['current_savings']:,.0f} TWD")
    print(f"月收入: {test_case['monthly_income']:,.0f} TWD")
    
    try:
        # 基礎版本
        basic_advisor = dspy.Predict(RetirementAdvice)
        basic_result = basic_advisor(**test_case)
        
        print(f"\n📋 基礎版本結果:")
        print(f"風險評估: {basic_result.risk_assessment}")
        print(f"建議: {basic_result.advice}")
        
        # ChainOfThought 版本
        cot_advisor = dspy.ChainOfThought(RetirementAdvice)
        cot_result = cot_advisor(**test_case)
        
        print(f"\n🧠 ChainOfThought 版本結果:")
        print(f"風險評估: {cot_result.risk_assessment}")
        print(f"建議: {cot_result.advice}")
        
        if hasattr(cot_result, 'reasoning'):
            print(f"推理過程: {cot_result.reasoning[:150]}...")
        
        print("\n🔍 對比分析:")
        print("- ChainOfThought 版本通常提供更詳細和有邏輯的分析")
        print("- 推理過程讓建議更有說服力")
        print("- 可以追蹤 AI 如何得出結論")
        
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        print("請確保 API key 正確設置")

def demo_optimization_concept():
    """展示優化概念"""
    print("\n\n=== 📚 Few-Shot 優化概念 ===\n")
    
    print("🎯 BootstrapFewShot 優化的工作原理:")
    
    print("\n1️⃣ 準備訓練範例:")
    print("```python")
    print("examples = [")
    print("    dspy.Example(")
    print("        current_age=30, current_savings=1000000, monthly_income=60000,")
    print("        risk_assessment=\"High\", advice=\"需要大幅增加儲蓄\"")
    print("    ),")
    print("    dspy.Example(")
    print("        current_age=50, current_savings=8000000, monthly_income=120000,")
    print("        risk_assessment=\"Low\", advice=\"儲蓄狀況良好\"")
    print("    )")
    print("]")
    print("```")
    
    print("\n2️⃣ 優化前的 Prompt:")
    print("```")
    print("Current Age: 35")
    print("Current Savings: 2000000.0")
    print("Monthly Income: 80000.0")
    print("Risk Assessment: [AI 生成]")
    print("Advice: [AI 生成]")
    print("```")
    
    print("\n3️⃣ 優化後的 Prompt (自動添加範例):")
    print("```")
    print("Current Age: 30")
    print("Current Savings: 1000000.0")
    print("Monthly Income: 60000.0")
    print("Risk Assessment: High")
    print("Advice: 需要大幅增加儲蓄")
    print("")
    print("Current Age: 50")
    print("Current Savings: 8000000.0")
    print("Monthly Income: 120000.0")
    print("Risk Assessment: Low")
    print("Advice: 儲蓄狀況良好")
    print("")
    print("Current Age: 35")
    print("Current Savings: 2000000.0")
    print("Monthly Income: 80000.0")
    print("Risk Assessment: [基於範例學習的更準確預測]")
    print("Advice: [基於類似案例的更具體建議]")
    print("```")
    
    print("\n✨ 優化的效果:")
    print("1. 📖 學習模式: 從範例中學習回答的風格和準確性")
    print("2. 🎯 上下文學習: 根據相似案例提供更準確的建議")
    print("3. 📈 自動選擇: 系統自動選擇最相關的範例")
    print("4. 🔄 動態優化: 根據新資料持續改進")

def demo_prompt_engineering_comparison():
    """對比傳統 prompt 工程和 dspy"""
    print("\n\n=== ⚖️ 傳統 vs dspy Prompt 工程 ===\n")
    
    print("❌ 傳統手動 Prompt 工程:")
    print("```python")
    print("prompt = '''")
    print("你是一位專業的退休規劃顧問。請根據以下資訊提供建議：")
    print("")
    print("年齡：{age} 歲")
    print("存款：{savings} TWD")
    print("月收入：{income} TWD")
    print("")
    print("請提供：")
    print("1. 風險評估 (低/中/高)")
    print("2. 具體建議")
    print("")
    print("請按以下格式回答：")
    print("風險評估：[你的評估]")
    print("建議：[你的建議]")
    print("'''")
    print("")
    print("response = call_openai(prompt.format(age=35, savings=2000000, income=80000))")
    print("# 需要手動解析回應...")
    print("# 需要處理格式不一致...")
    print("# 需要手動調整 prompt...")
    print("```")
    
    print("\n✅ dspy 自動化方式:")
    print("```python")
    print("class RetirementAdvice(dspy.Signature):")
    print("    age: int = dspy.InputField(desc=\"年齡\")")
    print("    savings: float = dspy.InputField(desc=\"存款 TWD\")")
    print("    income: float = dspy.InputField(desc=\"月收入 TWD\")")
    print("    ")
    print("    risk: str = dspy.OutputField(desc=\"風險評估\")")
    print("    advice: str = dspy.OutputField(desc=\"具體建議\")")
    print("")
    print("advisor = dspy.ChainOfThought(RetirementAdvice)")
    print("result = advisor(age=35, savings=2000000, income=80000)")
    print("# 自動格式化、解析、驗證")
    print("```")
    
    print("\n📊 對比優勢:")
    print("| 方面 | 傳統方式 | dspy 方式 |")
    print("|------|----------|-----------|")
    print("| Prompt 設計 | 手動撰寫 | 自動生成 |")
    print("| 格式一致性 | 需要手動確保 | 自動保證 |")
    print("| 錯誤處理 | 手動編寫 | 自動處理 |")
    print("| 優化 | 手動調整 | 自動優化 |")
    print("| 維護性 | 困難 | 簡單 |")
    print("| 類型安全 | 無 | 有 |")
    print("| 可重用性 | 低 | 高 |")

def main():
    """主程式"""
    print("🎯 dspy Prompt 轉換和優化完整展示")
    print("=" * 60)
    
    setup_dspy()
    
    # 展示各種轉換和優化
    demo_signature_transformation()
    demo_chain_of_thought_enhancement()
    demo_actual_execution()
    demo_optimization_concept()
    demo_prompt_engineering_comparison()
    
    print("\n" + "=" * 60)
    print("🎉 展示完成！")
    
    print("\n💡 dspy 的核心價值:")
    print("1. 🔄 自動轉換: Signature → 結構化 Prompt")
    print("2. 🧠 智能增強: ChainOfThought 自動推理")
    print("3. 📚 自動優化: Few-shot 範例學習")
    print("4. 🛡️ 類型安全: 自動驗證輸入輸出")
    print("5. 🔧 易維護: 修改定義即可更新 Prompt")
    print("6. 📈 持續改進: 基於資料自動優化效果")
    
    print(f"\n🔗 後續步驟:")
    print(f"1. 執行更多查詢來累積訓練資料")
    print(f"2. 使用 BootstrapFewShot 進行實際優化")
    print(f"3. 分析優化前後的效果差異")
    print(f"4. 建立專屬的退休規劃 AI 助手")

if __name__ == "__main__":
    main()