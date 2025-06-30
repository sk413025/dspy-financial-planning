"""
比較手動撰寫 prompt 與 dspy 自動生成的差異
"""
import os
import json

# ========================================
# 1. 傳統手動撰寫 Prompt 的方式
# ========================================
def manual_prompt_example():
    """傳統方式：需要手動設計 prompt"""
    
    # 手動撰寫的 prompt 模板
    manual_prompt = """
    Please analyze the following retirement planning query and extract the parameters.
    
    Query: {query}
    
    Extract the following information:
    - Years until retirement (as integer)
    - Expected annual return percentage (as float, e.g., 7.0 for 7%)
    - Annual volatility percentage (as float, e.g., 15.0 for 15%)
    - Annual spending in TWD (as float)
    - Current net worth in TWD (as float, default to 3000000 if not specified)
    - Annual inflation percentage (as float, default to 3.0 if not specified)
    - Maximum acceptable bankruptcy probability (as float, default to 5.0 if not specified)
    
    Format your response as JSON with keys: yrs, return_mu, return_sigma, spend, init_net, inflation, goal_pct
    
    Examples:
    Query: "If I retire in 25 years with 7% return and spend 1M annually"
    Response: {{"yrs": 25, "return_mu": 7.0, "return_sigma": 15.0, "spend": 1000000.0, "init_net": 3000000.0, "inflation": 3.0, "goal_pct": 5.0}}
    
    Query: "30 years to retirement, 6.5% ±15% returns, 1.2M yearly expenses, 5M saved"
    Response: {{"yrs": 30, "return_mu": 6.5, "return_sigma": 15.0, "spend": 1200000.0, "init_net": 5000000.0, "inflation": 3.0, "goal_pct": 5.0}}
    
    Now analyze the query and respond with JSON:
    """
    
    # 需要手動處理：
    # 1. 設計 prompt 結構
    # 2. 提供範例
    # 3. 指定輸出格式
    # 4. 處理邊界情況
    # 5. 解析 JSON 回應
    
    print("=== 傳統手動 Prompt 方式 ===")
    print("需要：")
    print("1. 手動撰寫詳細的 prompt 模板")
    print("2. 手動提供 few-shot 範例")
    print("3. 手動處理輸出格式")
    print("4. 手動解析和驗證結果")
    print("\nPrompt 長度:", len(manual_prompt), "字元")
    print("-" * 50)


# ========================================
# 2. dspy 自動生成 Prompt 的方式
# ========================================
def dspy_automatic_example():
    """dspy 方式：自動生成 prompt"""
    import dspy
    
    # 只需定義輸入輸出結構
    class QueryParser(dspy.Signature):
        """Parse natural language retirement query into structured parameters."""
        
        query: str = dspy.InputField(desc="Natural language retirement planning query")
        yrs: int = dspy.OutputField(desc="Years until retirement")
        return_mu: float = dspy.OutputField(desc="Expected annual return as percentage")
        return_sigma: float = dspy.OutputField(desc="Annual volatility as percentage")
        spend: float = dspy.OutputField(desc="Annual spending in TWD")
        init_net: float = dspy.OutputField(desc="Current net worth in TWD")
        inflation: float = dspy.OutputField(desc="Annual inflation as percentage")
        goal_pct: float = dspy.OutputField(desc="Maximum acceptable bankruptcy probability")
    
    print("\n=== dspy 自動生成 Prompt 方式 ===")
    print("只需要：")
    print("1. 定義 Signature 類別")
    print("2. 標註輸入輸出欄位")
    print("3. dspy 自動處理其餘部分")
    print("\n優勢：")
    print("- 不需手動撰寫 prompt")
    print("- 自動處理格式轉換")
    print("- 自動生成適當的指令")
    print("- 類型安全和驗證")
    print("-" * 50)


# ========================================
# 3. dspy 優化功能展示
# ========================================
def dspy_optimization_example():
    """展示 dspy 的優化功能"""
    import dspy
    from dspy.teleprompt import BootstrapFewShot
    
    print("\n=== dspy 優化功能 ===")
    
    # 定義任務
    class RetirementAdvice(dspy.Signature):
        """Generate retirement planning advice based on parameters."""
        
        bankruptcy_prob: float = dspy.InputField(desc="Bankruptcy probability percentage")
        years: int = dspy.InputField(desc="Years until retirement")
        advice: str = dspy.OutputField(desc="Personalized retirement advice")
    
    class RetirementAdvisor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generate_advice = dspy.ChainOfThought(RetirementAdvice)
        
        def forward(self, bankruptcy_prob, years):
            return self.generate_advice(bankruptcy_prob=bankruptcy_prob, years=years)
    
    # 準備訓練範例
    train_examples = [
        dspy.Example(
            bankruptcy_prob=80.0,
            years=30,
            advice="Your 80% bankruptcy risk is too high. Consider: 1) Increasing savings, 2) Reducing spending, 3) Working longer"
        ),
        dspy.Example(
            bankruptcy_prob=3.0,
            years=25,
            advice="Excellent! Your 3% bankruptcy risk is very low. Your retirement plan looks solid."
        ),
        dspy.Example(
            bankruptcy_prob=25.0,
            years=20,
            advice="Your 25% bankruptcy risk is concerning. Consider increasing your savings rate or adjusting expectations."
        ),
    ]
    
    print("優化功能包括：")
    print("\n1. BootstrapFewShot - 自動生成範例")
    print("   - 從少量範例中學習")
    print("   - 自動生成更多訓練資料")
    print("   - 優化 prompt 以提高準確度")
    
    print("\n2. 自動評估和改進")
    print("   - 定義評估指標")
    print("   - 自動測試不同 prompt 變體")
    print("   - 選擇最佳配置")
    
    print("\n3. 程式化優化流程：")
    print("```python")
    print("# 設定優化器")
    print("optimizer = BootstrapFewShot(metric=my_metric)")
    print("")
    print("# 編譯模組")
    print("compiled_advisor = optimizer.compile(")
    print("    RetirementAdvisor(),")
    print("    trainset=train_examples")
    print(")")
    print("")
    print("# 使用優化後的模組")
    print("result = compiled_advisor(bankruptcy_prob=15.0, years=25)")
    print("```")
    print("-" * 50)


# ========================================
# 4. 查看 dspy 自動生成的 prompt
# ========================================
def show_generated_prompt():
    """展示 dspy 實際生成的 prompt"""
    import dspy
    
    print("\n=== dspy 生成的 Prompt 內容 ===")
    
    # 配置一個簡單的任務
    class SimpleTask(dspy.Signature):
        """Calculate retirement risk assessment."""
        
        years: int = dspy.InputField(desc="Years to retirement")
        savings: float = dspy.InputField(desc="Current savings amount")
        risk_level: str = dspy.OutputField(desc="Risk level: Low, Medium, or High")
    
    # 創建預測器
    predictor = dspy.Predict(SimpleTask)
    
    # 查看生成的 prompt（這是概念展示）
    print("dspy 會自動生成類似這樣的 prompt：")
    print("\n```")
    print("Given the fields `years` and `savings`, produce the field `risk_level`.")
    print("")
    print("---")
    print("")
    print("Follow the following format.")
    print("")
    print("Years: Years to retirement")
    print("Savings: Current savings amount")  
    print("Risk Level: Risk level: Low, Medium, or High")
    print("")
    print("---")
    print("")
    print("Years: 25")
    print("Savings: 3000000.0")
    print("Risk Level: [OUTPUT]")
    print("```")
    
    print("\n關鍵差異：")
    print("1. 自動從 Signature 生成格式")
    print("2. 自動處理輸入輸出對應")
    print("3. 清晰的結構化格式")
    print("4. 不需要手動維護模板")
    print("-" * 50)


# ========================================
# 主程式
# ========================================
if __name__ == "__main__":
    print("DSPy 自動化 Prompt 工程示範")
    print("=" * 50)
    
    # 展示各種方式的差異
    manual_prompt_example()
    dspy_automatic_example()
    dspy_optimization_example()
    show_generated_prompt()
    
    print("\n總結：")
    print("1. dspy 讓你專注於定義任務，而非撰寫 prompt")
    print("2. 自動處理格式、解析、驗證等繁瑣工作")
    print("3. 提供優化工具來改進效果")
    print("4. 程式化的方式更容易維護和測試")