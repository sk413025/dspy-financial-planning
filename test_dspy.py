import dspy
import os

# Set API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("請設置 OPENAI_API_KEY 環境變數")
os.environ['OPENAI_API_KEY'] = api_key

# Try different ways to configure
try:
    # Method 1: Direct configure
    turbo = dspy.OpenAI(model='gpt-4o-mini')
    dspy.settings.configure(lm=turbo)
    print("Method 1 succeeded")
except Exception as e:
    print(f"Method 1 failed: {e}")

try:
    # Method 2: Using dspy.OpenAI
    from dspy import OpenAI
    turbo = OpenAI(model='gpt-4o-mini')
    dspy.settings.configure(lm=turbo)
    print("Method 2 succeeded")
except Exception as e:
    print(f"Method 2 failed: {e}")

try:
    # Method 3: Using dspy.models
    import dspy.models
    turbo = dspy.models.OpenAI(model='gpt-4o-mini')
    dspy.settings.configure(lm=turbo)
    print("Method 3 succeeded")
except Exception as e:
    print(f"Method 3 failed: {e}")

# Print available attributes
print("\nAvailable dspy attributes:")
print([attr for attr in dir(dspy) if not attr.startswith('_')])