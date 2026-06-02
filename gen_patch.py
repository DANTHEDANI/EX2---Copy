import os

patch_script = """
import os

files = {
    r"c:\\Users\\danie\\Documents\\VIBE_CODING\\EX2\\tests\\integration\\test_sdk.py": (
        "StockTradingEnv", "TradingEnv"
    ),
    r"c:\\Users\\danie\\Documents\\VIBE_CODING\\EX2\\tests\\unit\\test_training.py": (
        "StockTradingEnv", "TradingEnv"
    ),
    r"c:\\Users\\danie\\Documents\\VIBE_CODING\\EX2\\src\\trading_sdk\\services\\training.py": (
        "StockTradingEnv", "TradingEnv",
        "Conv1DDuelingDQN", "DuelingDQNNetwork"
    )
}

# Apply simple replace on the targeted broken imports
for file, targets in files.items():
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace("StockTradingEnv", "TradingEnv")
    content = content.replace("Conv1DDuelingDQN", "DuelingDQNNetwork")
    content = content.replace("model = Conv1DDuelingDQN(n_actions)", "model = DuelingDQNNetwork(action_dim=n_actions)")
    
    # Adjust initialization args inside training.py
    if "training.py" in file:
        content = content.replace(
            "env = TradingEnv(\\n            historical_data, \\n            initial_balance=self.config.setup[\\"environment\\"][\\"initial_balance\\"],\\n            commission=self.config.setup[\\"environment\\"][\\"commission_fee\\"]\\n        )",
            "env = TradingEnv(\\n            historical_data, \\n            initial_balance=self.config.setup[\\"environment\\"][\\"initial_balance\\"]\\n        )"
        )
        content = content.replace("action_dim=n_actions", "action_dim=n_actions")
        content = content.replace("model = DuelingDQNNetwork(n_actions)", "model = DuelingDQNNetwork(action_dim=n_actions)")

    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)

print("Patch imports applied.")
"""

with open(r"c:\Users\danie\Documents\VIBE_CODING\EX2\patch_tests.py", "w") as f:
    f.write(patch_script)
