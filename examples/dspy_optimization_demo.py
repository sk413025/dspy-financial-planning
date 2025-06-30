"""
實際展示 dspy 優化功能
"""
import dspy
import os

# 設定 API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")

# ========================================
# 實際優化範例
# ========================================

# 1. 定義任務：財務建議生成
class FinancialAdvice(dspy.Signature):
    """Generate personalized financial advice based on retirement metrics."""
    
    bankruptcy_prob: float = dspy.InputField(desc="Bankruptcy probability (%)")
    years_to_retire: int = dspy.InputField(desc="Years until retirement")
    current_savings: float = dspy.InputField(desc="Current savings (TWD)")
    annual_spend: float = dspy.InputField(desc="Annual spending (TWD)")
    
    risk_assessment: str = dspy.OutputField(desc="Risk level: Low/Medium/High")
    recommendations: str = dspy.OutputField(desc="Top 3 actionable recommendations")
    confidence_score: float = dspy.OutputField(desc="Confidence in retirement plan (0-100)")


# 2. 創建模組
class FinancialAdvisor(dspy.Module):
    def __init__(self):
        super().__init__()
        # 使用 ChainOfThought 來生成推理過程
        self.advisor = dspy.ChainOfThought(FinancialAdvice)
    
    def forward(self, bankruptcy_prob, years_to_retire, current_savings, annual_spend):
        return self.advisor(
            bankruptcy_prob=bankruptcy_prob,
            years_to_retire=years_to_retire,
            current_savings=current_savings,
            annual_spend=annual_spend
        )


# 3. 準備訓練資料（示範用）
def create_training_examples():
    """創建訓練範例"""
    examples = [
        dspy.Example(
            bankruptcy_prob=5.0,
            years_to_retire=30,
            current_savings=5000000.0,
            annual_spend=1000000.0,
            risk_assessment="Low",
            recommendations="1. Continue current strategy, 2. Consider diversifying investments, 3. Review plan annually",
            confidence_score=85.0
        ),
        dspy.Example(
            bankruptcy_prob=40.0,
            years_to_retire=20,
            current_savings=2000000.0,
            annual_spend=1500000.0,
            risk_assessment="High",
            recommendations="1. Increase savings rate urgently, 2. Reduce annual spending by 20%, 3. Consider delaying retirement",
            confidence_score=35.0
        ),
        dspy.Example(
            bankruptcy_prob=15.0,
            years_to_retire=25,
            current_savings=3500000.0,
            annual_spend=1200000.0,
            risk_assessment="Medium",
            recommendations="1. Boost savings by 10%, 2. Explore higher yield investments, 3. Build emergency fund",
            confidence_score=65.0
        ),
    ]
    return examples


# 4. 優化示範
def demonstrate_optimization():
    """展示優化過程"""
    print("=== DSPy 優化示範 ===\n")
    
    # 檢查 API key
    if not os.getenv("OPENAI_API_KEY"):
        print("注意：需要設定 OPENAI_API_KEY 才能執行實際優化")
        print("export OPENAI_API_KEY='your-key-here'\n")
        return
    
    try:
        # 配置語言模型
        lm = dspy.LM(model='openai/gpt-4o-mini', max_tokens=300)
        dspy.configure(lm=lm)
        
        # 創建基礎模組
        advisor = FinancialAdvisor()
        
        # 測試未優化版本
        print("1. 未優化版本測試：")
        print("-" * 30)
        
        test_case = {
            "bankruptcy_prob": 25.0,
            "years_to_retire": 22,
            "current_savings": 2800000.0,
            "annual_spend": 1100000.0
        }
        
        result = advisor(**test_case)
        print(f"輸入: 破產率={test_case['bankruptcy_prob']}%, 退休年數={test_case['years_to_retire']}")
        print(f"結果: 風險評估={result.risk_assessment}")
        print(f"建議: {result.recommendations}")
        print(f"信心分數: {result.confidence_score}")
        
        # 使用 BootstrapFewShot 優化
        print("\n2. 使用 BootstrapFewShot 優化：")
        print("-" * 30)
        
        from dspy.teleprompt import BootstrapFewShot
        
        # 定義評估指標
        def advice_metric(example, prediction, trace=None):
            # 檢查是否有所有必要欄位
            return (prediction.risk_assessment is not None and 
                    prediction.recommendations is not None and
                    prediction.confidence_score is not None)
        
        # 設定優化器
        optimizer = BootstrapFewShot(
            metric=advice_metric,
            max_bootstrapped_demos=3,
            max_labeled_demos=3
        )
        
        # 編譯優化版本
        train_examples = create_training_examples()
        compiled_advisor = optimizer.compile(
            advisor,
            trainset=train_examples[:2],  # 使用前兩個作為訓練
            valset=train_examples[2:]     # 最後一個作為驗證
        )
        
        # 測試優化版本
        print("優化後測試相同案例：")
        optimized_result = compiled_advisor(**test_case)
        print(f"優化後風險評估: {optimized_result.risk_assessment}")
        print(f"優化後建議: {optimized_result.recommendations}")
        print(f"優化後信心分數: {optimized_result.confidence_score}")
        
    except Exception as e:
        print(f"執行時發生錯誤: {e}")
        print("請確保已正確設定 OPENAI_API_KEY")


# 5. 展示 Prompt 比較
def show_prompt_comparison():
    """展示優化前後的 prompt 差異"""
    print("\n\n=== Prompt 優化比較 ===\n")
    
    print("優化前（基礎版）：")
    print("- 使用通用的 prompt 模板")
    print("- 沒有具體範例引導")
    print("- 依賴模型的一般理解")
    
    print("\n優化後（BootstrapFewShot）：")
    print("- 自動加入相關範例")
    print("- 學習訓練資料的模式")
    print("- 生成更準確的輸出格式")
    
    print("\n優化技術：")
    print("1. Few-shot learning: 從範例中學習")
    print("2. Bootstrap: 自動生成更多訓練資料")
    print("3. Metric-based: 根據評估指標優化")
    print("4. Automatic prompt engineering: 自動調整 prompt")


if __name__ == "__main__":
    demonstrate_optimization()
    show_prompt_comparison()
    
    print("\n\n總結：")
    print("1. dspy 自動處理 prompt 工程的複雜性")
    print("2. 優化器可以從少量範例中學習")
    print("3. 不需要手動調整 prompt")
    print("4. 可以根據特定指標進行優化")