import pandas as pd
from pathlib import Path

MONTH_NAMES = ["January","February","March","April","May","June","July","August","September","October","November","December"]

class SalesData:
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        # Try to normalize columns from variants like
        # ["sales.monthly","datum","Product","Sales"] or already clean.
        cols = [c.strip().lower() for c in df.columns.tolist()]
        if len(cols) >= 3:
            # Assume first col = date, second = product, third = sales
            df.rename(columns={
                df.columns[0]:"date",
                df.columns[1]:"Product",
                df.columns[2]:"Sales"
            }, inplace=True)

        if "date" not in df.columns:  # fallback if named 'datum'
            for c in df.columns:
                if c.lower() in ["datum","date","sales.monthly"]:
                    df.rename(columns={c:"date"}, inplace=True)

        self.df = df
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df["Product"] = self.df["Product"].astype(str).str.upper()
        self.df["Sales"] = pd.to_numeric(self.df["Sales"], errors="coerce").fillna(0.0)
        self.df["Year"] = self.df["date"].dt.year
        self.df["Month"] = self.df["date"].dt.month
        self.df["MonthName"] = self.df["Month"].apply(lambda m: MONTH_NAMES[m-1])

    def sales_on(self, year:int=None, month:str=None, product:str=None):
        q = self.df
        if year is not None:  q = q[q["Year"]==int(year)]
        if month:             q = q[q["MonthName"].str.lower()==month.lower()]
        if product:           q = q[q["Product"]==product.upper()]
        val = float(q["Sales"].sum()) if len(q) else 0.0
        return {"rows": int(len(q)), "total_sales": round(val, 2)}

    def top_products(self, year:int=None, k:int=5):
        q = self.df if year is None else self.df[self.df["Year"]==int(year)]
        agg = q.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(k)
        return [{"product":p, "sales": round(float(v),2)} for p,v in agg.items()]

    def peak_month(self, product:str):
        p = product.upper()
        q = self.df[self.df["Product"]==p]
        if q.empty: return {"product":p, "peak":"n/a", "sales":0}
        grp = q.groupby(["Year","MonthName"])["Sales"].sum()
        (yr, mn) = grp.idxmax()
        val = grp.max()
        return {"product":p, "peak": f"{mn} {yr}", "sales": round(float(val),2)}

    def yoy_mom(self, product:str, year:int, month:str):
        p = product.upper()
        m = MONTH_NAMES.index(month.capitalize()) + 1
        q = self.df[self.df["Product"]==p]
        if q.empty: return {"product":p, "note":"no data"}
        cur = q[(q["Year"]==int(year)) & (q["Month"]==m)]["Sales"].sum()

        last_m = 12 if m==1 else m-1
        last_m_year = int(year)-1 if m==1 else int(year)
        lm = q[(q["Year"]==last_m_year) & (q["Month"]==last_m)]["Sales"].sum()
        ly = q[(q["Year"]==int(year)-1) & (q["Month"]==m)]["Sales"].sum()

        def pct(a,b): return None if b==0 else round(100*(a-b)/b,2)
        return {
            "product": p, "year": int(year), "month": month.capitalize(),
            "value": round(float(cur),2),
            "vs_last_month_pct": pct(cur,lm),
            "vs_last_year_pct": pct(cur,ly)
        }
