# DSPy Financial Planning System

一個使用 DSPy 框架構建的完整金融規劃系統，展示了如何透過聲明式程式設計來自動化 prompt 生成與優化。

## 💡 為什麼選擇這個專案？

**傳統 LLM 應用開發的痛點：**
- 手動撰寫和調整 prompt，耗時且容易出錯
- 難以系統化地優化 prompt 效果
- 缺乏結構化的輸入輸出驗證
- 實驗記錄和結果比較困難

**DSPy 如何解決這些問題：**
- 🎯 **聲明式設計**: 只需定義輸入輸出結構，prompt 自動生成
- 🧠 **智能優化**: 自動添加推理步驟並基於範例學習優化
- 📊 **系統化比較**: 內建三種方法比較（Predict vs ChainOfThought vs Few-Shot）
- 📝 **完整記錄**: 自動記錄所有實驗數據供分析

## 🎯 本專案展示內容

### 1. DSPy 核心概念
- **Signature 定義**: 如何用類型提示定義 LLM 任務
- **自動 Prompt 生成**: DSPy 如何將 Signature 轉換為結構化 prompt
- **三種優化方法**: 從基礎到進階的自動優化技術

### 2. 實際應用場景
- **退休風險評估**: 使用 Monte Carlo 模擬進行財務規劃
- **自然語言查詢**: 將人類語言轉換為結構化參數
- **專家建議生成**: 根據財務狀況提供個人化建議

### 3. 完整開發流程
- **實驗追蹤**: 詳細記錄每次實驗的參數和結果
- **效果比較**: 量化分析不同方法的表現差異
- **可視化分析**: 圖表展示模擬結果和趨勢

## 🚀 快速開始

### 步驟 1: 環境設置

```bash
# 複製專案
git clone https://github.com/sk413025/dspy-financial-planning.git
cd dspy-financial-planning

# 安裝依賴
pip install dspy-ai numpy matplotlib pandas

# 設置 API Key
cp config.example.py config.py
# 編輯 config.py，填入您的 OpenAI API key
```

### 步驟 2: 體驗 DSPy Prompt 優化

運行主要演示，看看 DSPy 如何自動生成和優化 prompt：

```bash
python demo_prompt_optimization.py
```

**您將看到：**
```
🎯 dspy Prompt 優化完整實戰演示
================================================================================
本次演示將展示：
1. 📋 基礎 Prompt 自動生成
2. 🧠 ChainOfThought 推理增強  
3. 📚 Few-Shot 範例學習優化
4. 📊 三種方式的結果對比
================================================================================

✅ 基礎版本結果:
🎯 風險等級: 中風險
💰 建議儲蓄: 建議每月儲蓄約 2.5 萬台幣
📋 策略: 考慮將資金分散投資於股票和債券，以平衡風險和回報...

🧠 ChainOfThought 版本結果:
🎯 風險等級: 中風險  
💰 建議儲蓄: 建議每月儲蓄金額約為3萬台幣
📋 策略: 建議您將每月的儲蓄金額設定為3萬台幣...

🧠 推理過程:
您目前35歲，擁有200萬台幣的存款，月收入為8萬台幣，目標退休年齡為65歲...

📊 完整演示日誌已保存到: logs/session_20250630_185304.jsonl
```

### 步驟 3: 自然語言查詢

用自然語言提問，系統會自動解析並運行 Monte Carlo 模擬：

```bash
python pipeline/run.py "如果我25年後退休，期望報酬率6%，波動度12%，每年花費80萬，目前有200萬存款，破產機率是多少？"
```

**結果：**
```
Parsing query: 如果我25年後退休，期望報酬率6%，波動度12%...

Parsed parameters:
  yrs: 25
  return_mu: 6.0
  return_sigma: 12.0
  spend: 800000.0
  init_net: 2000000.0

Running Monte Carlo simulation...

Results:
{
  "bankruptcy_probability": 89.2,
  "meets_goal": false,
  "final_balance_mean": 1250000.0
}
```

### 步驟 4: 查看詳細實驗記錄

使用我們的日誌查看工具來分析實驗結果：

```bash
python view_logs.py
```

這會顯示所有可用的日誌檔案，然後：

```bash
python view_logs.py session_20250630_185304.jsonl
```

**您將看到完整的結構化報告：**
- 📝 Signature 定義和自動生成的 prompt
- 🧪 三種方法的測試參數和結果
- 📚 Few-Shot 訓練範例
- 📊 詳細的結果比較分析
- 💡 關鍵洞察和建議

## 🔬 深入理解 DSPy

### DSPy Signature: 任務定義的核心

```python
class RetirementRisk(dspy.Signature):
    """評估退休風險並提供建議"""
    
    # 輸入欄位
    age: int = dspy.InputField(desc="當前年齡")
    savings: float = dspy.InputField(desc="當前存款金額(萬台幣)")
    monthly_income: float = dspy.InputField(desc="月收入(台幣)")
    
    # 輸出欄位  
    risk_level: str = dspy.OutputField(desc="風險等級: 低風險/中風險/高風險")
    monthly_save_needed: str = dspy.OutputField(desc="建議每月儲蓄金額")
    strategy: str = dspy.OutputField(desc="具體退休策略建議")
```

**DSPy 自動生成的 Prompt:**
```
評估退休風險並提供建議

---

Follow the following format.

Age: 當前年齡
Savings: 當前存款金額(萬台幣)
Monthly Income: 月收入(台幣)
Risk Level: 風險等級: 低風險/中風險/高風險
Monthly Save Needed: 建議每月儲蓄金額
Strategy: 具體退休策略建議

---

Age: 35
Savings: 200.0
Monthly Income: 80000
Risk Level:
```

### 三種優化方法比較

| 方法 | 特點 | 適用場景 | 結果品質 |
|------|------|----------|----------|
| **Predict** | 簡單直接 | 快速原型 | 基礎 |
| **ChainOfThought** | 自動推理 | 複雜邏輯 | 更好 |
| **Few-Shot** | 範例學習 | 一致性要求 | 最佳 |

### 實驗記錄系統

所有實驗都會自動記錄到 JSON 格式，包含：

```json
{
  "id": 1,
  "timestamp": "2025-06-30T18:53:04",
  "intermediate": {
    "steps": [/* 執行步驟 */],
    "parameters": {/* 測試參數 */},
    "predictions": [/* 預測結果 */],
    "training_examples": {/* 訓練範例 */},
    "comparison": {/* 結果比較 */}
  },
  "output": {
    "demo_summary": {/* 總結分析 */}
  }
}
```

## 📁 專案結構詳解

```
├── config.example.py          # API key 配置範本
├── demo_prompt_optimization.py # 主要演示程式
├── view_logs.py               # 日誌查看工具
│
├── examples/                  # DSPy 範例程式
│   ├── dspy_prompt_engineering.py    # Prompt 工程技術
│   ├── inspect_dspy_prompts.py       # Prompt 內部結構
│   └── prompt_transformation_demo.py # 轉換過程展示
│
├── finance/
│   └── core.py               # 財務數據結構
│
├── pipeline/
│   └── run.py                # 自然語言查詢處理器
│
├── prompts/
│   └── retire.py             # 退休規劃 DSPy 模組
│
├── simulation/
│   └── monte_carlo.py        # Monte Carlo 模擬引擎
│
├── utils/
│   ├── logger.py             # 實驗記錄系統
│   └── visualizer.py         # 數據可視化工具
│
└── logs/                     # 自動生成的實驗記錄
    ├── session_*.jsonl       # 詳細實驗日誌
    └── summary_*.json        # 總結報告
```

## 🎯 學習路徑建議

### 初學者
1. 先運行 `demo_prompt_optimization.py` 了解基本概念
2. 查看 `examples/inspect_dspy_prompts.py` 理解 Prompt 生成機制
3. 嘗試修改 Signature 定義，觀察 Prompt 變化

### 進階使用者
1. 研究 `pipeline/run.py` 了解完整的應用流程
2. 分析 `logs/` 中的實驗記錄，理解不同方法的效果差異
3. 使用 `utils/visualizer.py` 創建自己的分析圖表

### 開發者
1. 擴展 `prompts/retire.py` 添加新的 Signature
2. 修改 `utils/logger.py` 增加自定義的記錄功能
3. 集成到您自己的 LLM 應用中

## 🔧 進階功能

### 自定義評估指標

```python
def custom_metric(example, prediction, trace=None):
    """自定義評估函數"""
    # 檢查風險等級準確性
    risk_correct = prediction.risk_level == example.risk_level
    
    # 檢查建議合理性
    advice_quality = len(prediction.strategy) > 20
    
    return int(risk_correct and advice_quality)
```

### 批量實驗分析

```python
from utils.logger import ExperimentLogger

# 載入多個實驗記錄
logger = ExperimentLogger()
df = logger.get_dataframe()

# 分析趨勢
success_rate = df['success'].mean()
avg_duration = df['duration_ms'].mean()
```

## 🤝 貢獻指南

1. Fork 這個專案
2. 創建功能分支: `git checkout -b feature-name`
3. 提交變更: `git commit -am 'Add feature'`
4. 推送到分支: `git push origin feature-name`
5. 提交 Pull Request

## 📚 相關資源

- [DSPy 官方文檔](https://github.com/stanfordnlp/dspy)
- [Monte Carlo 模擬原理](https://en.wikipedia.org/wiki/Monte_Carlo_method)
- [OpenAI API 文檔](https://platform.openai.com/docs)

## 🎉 總結

這個專案展示了 DSPy 如何革命性地簡化 LLM 應用開發：

- ✅ **零手動 Prompt**: 只需定義結構，Prompt 自動生成
- ✅ **自動優化**: 系統化地改進 LLM 表現
- ✅ **完整追蹤**: 每個實驗都有詳細記錄
- ✅ **實用案例**: 真實的金融規劃應用場景

**立即開始體驗 DSPy 的強大功能吧！** 🚀

---

使用 ❤️ 和 [DSPy](https://github.com/stanfordnlp/dspy) 構建