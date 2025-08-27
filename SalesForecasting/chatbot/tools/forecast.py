import pickle, plotly.graph_objects as go
from pathlib import Path
import pandas as pd

class Forecaster:
    def __init__(self, model_dir:Path, sales_df:pd.DataFrame):
        self.model_dir = Path(model_dir)
        self.sales_df = sales_df.copy()

    def forecast(self, product:str, periods:int=12, save_dir:Path=Path("./server/static/forecast")):
        save_dir.mkdir(parents=True, exist_ok=True)
        p = self.model_dir / f"{product.upper()}.pkl"
        if not p.exists():
            return {"ok": False, "error": f"model not found for {product}"}

        model = pickle.load(open(p, "rb"))  # SARIMAXResults
        ts = (self.sales_df[self.sales_df["Product"]==product.upper()]
                .sort_values("date")[["date","Sales"]])
        ts = ts.set_index("date")["Sales"].asfreq("M")

        pred = model.get_forecast(steps=periods)
        fc = pred.predicted_mean
        ci = pred.conf_int(alpha=0.2)

        fig = go.Figure()
        fig.add_scatter(x=ts.index, y=ts.values, mode="lines", name="History")
        fig.add_scatter(x=fc.index, y=fc.values, mode="lines", name="Forecast")
        fig.add_scatter(x=ci.index, y=ci.iloc[:,0], mode="lines", name="Lower", line=dict(dash="dot"))
        fig.add_scatter(x=ci.index, y=ci.iloc[:,1], mode="lines", name="Upper", line=dict(dash="dot"))

        out = save_dir / f"{product.upper()}_{periods}.html"
        fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
        return {"ok": True, "html_url": f"/static/forecast/{out.name}"}
