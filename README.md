# DSPy Financial Planning System

A comprehensive financial planning system built with the DSPy framework, featuring Monte Carlo simulations, natural language processing, and automatic prompt optimization for retirement risk assessment.

## 🚀 Features

- **Natural Language Query Processing**: Ask retirement planning questions in plain English
- **Monte Carlo Simulations**: Advanced retirement risk modeling with 10,000+ simulations
- **DSPy Prompt Optimization**: Demonstrates Predict vs ChainOfThought vs Few-Shot learning
- **Comprehensive Logging**: Track all inputs, outputs, and intermediate results
- **Visualization Tools**: Generate charts and reports from simulation data
- **Multiple Demo Examples**: Various ways to explore DSPy capabilities

## 📁 Project Structure

```
├── config.example.py          # Configuration template
├── demo_prompt_optimization.py # Main DSPy demonstration
├── examples/                  # Additional DSPy examples
│   ├── dspy_prompt_engineering.py
│   ├── inspect_dspy_prompts.py
│   └── prompt_transformation_demo.py
├── finance/
│   └── core.py               # Financial data structures
├── pipeline/
│   └── run.py                # Natural language query processor
├── prompts/
│   └── retire.py             # DSPy signatures and modules
├── simulation/
│   └── monte_carlo.py        # Monte Carlo simulation engine
├── utils/
│   ├── logger.py             # Comprehensive logging system
│   └── visualizer.py         # Data visualization tools
└── logs/                     # Session logs and reports
```

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sk413025/dspy-financial-planning.git
cd dspy-financial-planning
```

### 2. Install Dependencies

```bash
pip install dspy-ai numpy matplotlib pandas
```

### 3. Configure API Key

```bash
# Copy the example config file
cp config.example.py config.py

# Edit config.py and add your OpenAI API key
# OPENAI_API_KEY = "your-api-key-here"
```

## 🎯 Quick Start

### Basic DSPy Demonstration

Run the main demonstration to see DSPy's prompt optimization in action:

```bash
python demo_prompt_optimization.py
```

**Expected Output:**
```
🎯 dspy Prompt 優化完整實戰演示
================================================================================
本次演示將展示：
1. 📋 基礎 Prompt 自動生成
2. 🧠 ChainOfThought 推理增強
3. 📚 Few-Shot 範例學習優化
4. 📊 三種方式的結果對比
================================================================================
✅ dspy 已配置完成

================================================================================
📋 第一步: 基礎 Prompt 結構
================================================================================

🔍 我們定義的 Signature:
```python
class RetirementRisk(dspy.Signature):
    """評估退休風險並提供建議"""
    age: int = dspy.InputField(desc="當前年齡")
    savings: float = dspy.InputField(desc="當前存款金額(萬台幣)")
    # ... more fields
```

✅ 基礎版本結果:
🎯 風險等級: 中風險
💰 建議儲蓄: 建議每月儲蓄約 2.5 萬台幣
📋 策略: 考慮將資金分散投資於股票和債券，以平衡風險和回報...

🧠 推理過程:
您目前35歲，擁有200萬台幣的存款，月收入為8萬台幣，目標退休年齡為65歲...
```

### Natural Language Query Processing

Ask retirement planning questions in natural language:

```bash
python pipeline/run.py "If I retire in 25 years with 6% return and 12% volatility, spending 800k TWD annually with 2M TWD saved, what's my bankruptcy risk?"
```

**Expected Output:**
```
Parsing query: If I retire in 25 years with 6% return and 12% volatility...

Parsed parameters:
  yrs: 25
  return_mu: 6.0
  return_sigma: 12.0
  spend: 800000.0
  init_net: 2000000.0

Running Monte Carlo simulation...

Results:
{
  "bankruptcy_probability": 100.0,
  "meets_goal": false,
  "final_balance_mean": 0.0,
  "parameters": {
    "yrs": 25,
    "return_mu": 6.0,
    "return_sigma": 12.0,
    "spend": 800000.0,
    "init_net": 2000000.0
  }
}

📊 Session logged to: logs/session_20250630_184320.jsonl
```

## 📚 Examples

### 1. DSPy Prompt Engineering

Explore how DSPy automatically generates and optimizes prompts:

```bash
python examples/dspy_prompt_engineering.py
```

### 2. Inspect DSPy Prompts

See the internal structure of automatically generated prompts:

```bash
python examples/inspect_dspy_prompts.py
```

**Sample Output:**
```
=== 🔍 實際檢視 dspy 生成的 Prompt ===

🤖 dspy 自動生成的完整 Prompt:
```
Provide retirement advice.

---

Follow the following format.

Age: Age in years
Savings: Savings in TWD
Advice: Retirement advice

---

Age: 45
Savings: 5000000.0
Advice:
```

### 3. Prompt Transformation Demo

See how DSPy transforms signatures into structured prompts:

```bash
python examples/prompt_transformation_demo.py
```

## 🔬 Core Components

### DSPy Signatures

The system uses DSPy signatures to define structured inputs and outputs:

```python
class RetirementRisk(dspy.Signature):
    """評估退休風險並提供建議"""
    
    age: int = dspy.InputField(desc="當前年齡")
    savings: float = dspy.InputField(desc="當前存款金額(萬台幣)")
    monthly_income: float = dspy.InputField(desc="月收入(台幣)")
    
    risk_level: str = dspy.OutputField(desc="風險等級: 低風險/中風險/高風險")
    monthly_save_needed: str = dspy.OutputField(desc="建議每月儲蓄金額")
    strategy: str = dspy.OutputField(desc="具體退休策略建議")
```

### Monte Carlo Simulation

Advanced financial modeling with:
- 365-day daily compounding
- Log-normal return distribution
- Inflation-adjusted spending
- Configurable parameters

```python
def run_sim(mu: float, sigma: float, yrs: int, init_net: float, 
           spend: float, inflation: float, n: int = 10000) -> dict:
    # Monte Carlo simulation with daily returns
    daily_mu = mu / 365
    daily_sigma = sigma / np.sqrt(365)
    # ... simulation logic
```

### Comprehensive Logging

All interactions are logged with detailed metadata:

```python
# Session logging
logger.start_query(query, source="cli")
logger.log_parameters(parsed_params)
logger.log_simulation_results(results)
logger.save_entry()
```

## 📊 DSPy Optimization Techniques

### 1. Basic Predict
Simple input-output mapping without reasoning steps.

### 2. Chain of Thought
Automatically adds reasoning steps to improve accuracy:
```
Reasoning: Let's think step by step in order to 評估退休風險並提供建議.
```

### 3. Few-Shot Learning
Uses example-based learning to improve consistency and quality:
```python
training_examples = [
    dspy.Example(
        age=25, savings=50.0, monthly_income=50000,
        risk_level="高風險",
        monthly_save_needed="至少需要每月存2萬元"
    ),
    # ... more examples
]
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (set in config.py)

### DSPy Settings
- `DSPY_MODEL`: "openai/gpt-4o-mini" (configurable)
- `DSPY_MAX_TOKENS`: 500 (configurable)

## 📈 Results Analysis

The system provides comprehensive analysis including:

- **Risk Assessment**: Low/Medium/High risk classification
- **Savings Recommendations**: Specific monthly saving amounts
- **Investment Strategies**: Tailored advice based on age and situation
- **Reasoning Transparency**: Full reasoning process visibility

### Comparison Results

| Method | Risk Level | Savings Advice | Reasoning Quality |
|--------|------------|----------------|-------------------|
| Basic Predict | 中風險 | 2.5萬/月 | Simple, direct |
| Chain of Thought | 中風險 | 3萬/月 | Detailed reasoning |
| Few-Shot | 中風險 | 3萬/月 | Example-based learning |

## 🎯 Key Benefits

1. **🔄 Automated Prompt Generation**: No manual prompt writing required
2. **📈 Progressive Optimization**: From simple to sophisticated automatically  
3. **🎯 Goal-Oriented**: Based on actual performance metrics
4. **🧠 Reasoning Transparency**: See how AI reaches conclusions
5. **📊 Comprehensive Logging**: Track all experiments and results
6. **🔧 Easy Maintenance**: Modify signatures to update behavior

## 🔒 Security

- API keys are stored in `config.py` which is git-ignored
- No sensitive data is committed to the repository
- Example configuration provided in `config.example.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🚀 Next Steps

1. **Add More Examples**: Expand the training dataset for Few-Shot learning
2. **Real Data Integration**: Connect with actual financial data sources
3. **Advanced Metrics**: Implement custom evaluation metrics
4. **UI Development**: Build a web interface for easier interaction
5. **Model Comparison**: Test different language models and compare results

---

Built with ❤️ using [DSPy](https://github.com/stanfordnlp/dspy) - The framework for programming with language models.