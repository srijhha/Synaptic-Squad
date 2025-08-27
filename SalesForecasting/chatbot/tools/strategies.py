import json
from pathlib import Path

class Strategies:
    def __init__(self, rules_path:Path):
        self.rules = json.loads(Path(rules_path).read_text())

    def suggest(self, trend:str=None, mom_pct=None, yoy_pct=None):
        tips=[]
        if trend=="up" or (mom_pct and mom_pct>5) or (yoy_pct and yoy_pct>5):
            tips.append(self.rules["high_growth"])
        if trend=="down" or (mom_pct and mom_pct<-5) or (yoy_pct and yoy_pct<-5):
            tips.append(self.rules["declining"])
        return tips or [self.rules["low_growth"]]
