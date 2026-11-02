import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

df = pd.DataFrame({
    "Experience": ["Beach", "Mountains", "Heritage"],
    "Budget": ["Low", "Medium", "High"],
})

# ---------- 1. Label Encoding ----------
le = LabelEncoder()
df["Exp_label"] = le.fit_transform(df["Experience"])

t1 = Table(title="1. Label Encoding", box=box.ROUNDED, header_style="bold cyan")
t1.add_column("Experience")
t1.add_column("Exp_label", justify="center", style="yellow")
for r in df.itertuples():
    t1.add_row(r.Experience, str(r.Exp_label))
console.print(t1)

# ---------- 2. Ordinal Encoding ----------
oe = OrdinalEncoder(categories=[["Low", "Medium", "High"]])
df["Budget_enc"] = oe.fit_transform(df[["Budget"]]).astype(int)

t2 = Table(title="2. Ordinal Encoding", box=box.ROUNDED, header_style="bold magenta")
t2.add_column("Budget")
t2.add_column("Budget_enc", justify="center", style="yellow")
for r in df.itertuples():
    t2.add_row(r.Budget, str(r.Budget_enc))
console.print(t2)

# ---------- 3. One-Hot Encoding ----------
ohe = pd.get_dummies(df["Experience"], dtype=int)

t3 = Table(title="3. One-Hot Encoding", box=box.ROUNDED, header_style="bold green")
t3.add_column("Experience")
for col in ohe.columns:
    t3.add_column(col, justify="center")
for name, (_, row) in zip(df["Experience"], ohe.iterrows()):
    cells = [f"[bold green]{v}[/bold green]" if v == 1 else f"[dim]{v}[/dim]"
             for v in row]
    t3.add_row(name, *cells)
console.print(t3)