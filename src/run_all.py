"""
Run-all pipeline for the Airbnb visual analytics project.

This module intentionally keeps logic simple and readable for portfolio review.
It loads the three dataset files, performs core cleaning/integration steps,
and generates the key figures into an output directory.

Usage:
    python -m src.run_all --data-dir data/raw --out-dir results/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def _normalize_columns(df):
    # strip whitespace, lower, replace spaces with underscores
    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df

def _coalesce(df, target, candidates):
    """Create df[target] from the first candidate column that exists."""
    for c in candidates:
        if c in df.columns:
            if target not in df.columns:
                df[target] = df[c]
            return df
    return df


def _parse_percent(series: pd.Series) -> pd.Series:
    """Convert '85%' -> 85.0, '0.85' -> 85.0 if likely fraction, else numeric."""
    s = series.astype(str).str.strip()
    s = s.str.replace("%", "", regex=False)
    s = pd.to_numeric(s, errors="coerce")
    # if values look like fractions (<=1), convert to percent
    s = np.where((s.notna()) & (s <= 1), s * 100, s)
    return pd.Series(s, index=series.index)

def _ensure_dir(path):
    from pathlib import Path
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save(fig, out_dir, filename, dpi=200):
    out_dir = _ensure_dir(out_dir)
    out_path = out_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")

def add_standard_columns(listings: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and create the canonical fields expected by the plots:
    price, room_type, neighbourhood, response_rate, amenities
    """
    df = listings.copy()

    # normalize already done in load_data(); safe to re-run if needed
    df.columns = df.columns.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)

    # room_type (your 'Room Type' -> room_type after normalization)
    if "room_type" not in df.columns:
        # fallback: find a close match
        for c in df.columns:
            if "room" in c and "type" in c:
                df["room_type"] = df[c]
                break

    # neighbourhood: prefer neighbourhood_cleansed
    if "neighbourhood" not in df.columns:
        if "neighbourhood_cleansed" in df.columns:
            df["neighbourhood"] = df["neighbourhood_cleansed"]
        else:
            # last fallback: any neighbourhood-like column
            for c in df.columns:
                if "neigh" in c:
                    df["neighbourhood"] = df[c]
                    break

    # price: auto-detect if not present
    if "price" not in df.columns:
        price_candidates = [c for c in df.columns if "price" in c]
        if price_candidates:
            df["price"] = df[price_candidates[0]]

    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # host response rate
    if "response_rate" not in df.columns:
        # normalized from host_response_rate
        if "host_response_rate" in df.columns:
            df["response_rate"] = _parse_percent(df["host_response_rate"])
        else:
            # attempt match
            for c in df.columns:
                if "response" in c and "rate" in c:
                    df["response_rate"] = _parse_percent(df[c])
                    break

    # amenities
    if "amenities" in df.columns:
        df["amenities"] = df["amenities"].astype(str)

    return df

def host_segment_from_counts(df: pd.DataFrame) -> pd.Series:
    """
    Segment hosts by portfolio size.
    Prefers host_total_listings_count; falls back to host_listings_count.
    """
    count_col = None
    for c in ["host_total_listings_count", "host_listings_count"]:
        if c in df.columns:
            count_col = c
            break

    if count_col is None:
        # if missing, treat all as small
        return pd.Series(["Small Portfolio (1 listing)"] * len(df), index=df.index)

    x = pd.to_numeric(df[count_col], errors="coerce").fillna(1)

    seg = pd.cut(
        x,
        bins=[0, 1, 5, 10, np.inf],
        labels=[
            "Small Portfolio (1 listing)",
            "Medium Portfolio (2–5 listings)",
            "Large Portfolio (6–10 listings)",
            "Enterprise Portfolio (10+ listings)",
        ],
        include_lowest=True,
        right=True,
    )
    return seg.astype(str)

def load_data(data_dir: str):
    from pathlib import Path
    import pandas as pd
    import geopandas as gpd

    data_dir = Path(data_dir)

    listings_path = data_dir / "listings.csv"
    reviews_path = data_dir / "reviews.csv"
    neigh_path = data_dir / "neighbourhoods.geojson"

    missing = [p.name for p in [listings_path, reviews_path, neigh_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required file(s) in {data_dir}: {', '.join(missing)}"
        )

    # 1) READ FIRST (critical)
    listings = pd.read_csv(listings_path)
    reviews = pd.read_csv(reviews_path)
    neighbourhoods = gpd.read_file(neigh_path)

    # 2) NORMALIZE COLUMN NAMES
    listings = _normalize_columns(listings)
    reviews = _normalize_columns(reviews)

    neighbourhoods = neighbourhoods.copy()
    neighbourhoods.columns = neighbourhoods.columns.astype(str).str.strip().str.lower()

    # 3) COALESCE COMMON FIELDS
    # room type: your "Room Type" becomes "room_type" after normalization
    listings = _coalesce(listings, "room_type", ["room_type"])

    # neighbourhood: prefer cleansed if available
    listings = _coalesce(listings, "neighbourhood", ["neighbourhood", "neighborhood", "neighbourhood_cleansed"])

    # price: detect likely price-like column names
    # (we will map the first match into "price")
    price_candidates = [c for c in listings.columns if "price" in c]
    if "price" not in listings.columns and price_candidates:
        listings["price"] = listings[price_candidates[0]]

    # 4) CLEAN PRICE INTO NUMERIC (if present)
    if "price" in listings.columns:
        listings["price"] = (
            listings["price"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        listings["price"] = pd.to_numeric(listings["price"], errors="coerce")

    return listings, reviews, neighbourhoods



def clean_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Basic review cleaning (date parsing + null handling)."""
    reviews = reviews.copy()
    if "date" in reviews.columns:
        reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    return reviews


def basic_clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Light-touch cleaning; extend with your notebook logic as needed."""
    listings = listings.copy()

    # Common Airbnb fields; keep robust to schema differences
    if "price" in listings.columns:
        # Price sometimes comes as strings like "$123.00"
        listings["price"] = (
            listings["price"]
            .astype(str)
            .str.replace(r"[^0-9.]", "", regex=True)
        )
        listings["price"] = pd.to_numeric(listings["price"], errors="coerce")

    return listings


def make_price_by_room_type(listings: pd.DataFrame, out_dir: Path) -> None:
    """Violin + box plot of price by room type (if columns exist)."""
    if not {"price", "room_type"}.issubset(listings.columns):
        return

    df = listings.dropna(subset=["price", "room_type"]).copy()
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    sns.violinplot(data=df, x="room_type", y="price", inner=None)
    sns.boxplot(data=df, x="room_type", y="price", width=0.2)
    plt.title("Price distribution by room type")
    plt.xlabel("Room type")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig01_price_by_room_type.png", dpi=200)
    plt.close()


def make_top_expensive_neighbourhoods(listings: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    """Bar chart: top-N neighbourhoods by median price (if columns exist)."""
    # column name may vary; adjust to match your dataset
    possible_cols = ["neighbourhood", "neighbourhood_cleansed", "neighbourhood_group"]
    ncol = next((c for c in possible_cols if c in listings.columns), None)
    if ncol is None or "price" not in listings.columns:
        return

    df = listings.dropna(subset=[ncol, "price"]).copy()
    if df.empty:
        return

    agg = (
        df.groupby(ncol)["price"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(10, 6))
    agg.plot(kind="barh")
    plt.title(f"Top {top_n} most expensive neighbourhoods (median price)")
    plt.xlabel("Median price")
    plt.ylabel("Neighbourhood")
    plt.tight_layout()
    plt.savefig(out_dir / "Fig02_top_expensive_neighbourhoods.png", dpi=200)
    plt.close()


def plot_price_by_room_type_violin(listings: pd.DataFrame, out_dir: str):
    df = listings.dropna(subset=["price", "room_type"]).copy()
    df = df[df["price"] > 0]

    # Trim extreme outliers so the plot + labels are readable (matches typical notebook approach)
    cap = df["price"].quantile(0.99)  # use 99th percentile cap
    df_trim = df[df["price"] <= cap].copy()

    room_order = df_trim["room_type"].value_counts().index.tolist()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    data = [df_trim.loc[df_trim["room_type"] == rt, "price"].dropna().values for rt in room_order]
    ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    ax.boxplot(data, positions=np.arange(1, len(room_order) + 1), widths=0.25)

    ax.set_title("Sydney Airbnb Price Distribution by Room Type")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Nightly Price ($)")
    ax.set_xticks(np.arange(1, len(room_order) + 1))
    ax.set_xticklabels(room_order)

    # Set y-limit to trimmed cap so labels sit in the visible region
    ax.set_ylim(0, cap * 1.05)

    # Medians from the FULL data (or trimmed—choose one; your screenshot looks like medians on trimmed)
    for i, rt in enumerate(room_order, start=1):
        med = np.nanmedian(df_trim.loc[df_trim["room_type"] == rt, "price"].values)
        ax.text(i, med, f"${int(round(med))}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    _save(fig, out_dir, "Fig07_price_violin_by_room_type.png")



def plot_reviews_by_day_of_week(reviews: pd.DataFrame, out_dir: str):
    df = reviews.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["day_of_week"] = df["date"].dt.day_name()

    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts = df["day_of_week"].value_counts().reindex(order).fillna(0).astype(int)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    bars = ax.bar(order, counts.values)

    ax.set_title("Figure 1: Number of Reviews by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Number of Reviews")
    ax.tick_params(axis="x", rotation=35)

    # add labels
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    _save(fig, out_dir, "Fig03_reviews_by_day_of_week.png")


def plot_weekday_weekend_share(reviews: pd.DataFrame, out_dir: str):
    df = reviews.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    dow = df["date"].dt.dayofweek  # Monday=0
    is_weekend = dow >= 5
    labels = ["Weekday", "Weekend"]
    values = [int((~is_weekend).sum()), int(is_weekend.sum())]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Figure 2: Share of Reviews:\nWeekday vs Weekend")

    _save(fig, out_dir, "Fig04_weekday_vs_weekend_pie.png")


def plot_correlation_heatmap(listings: pd.DataFrame, out_dir: str):
    df = listings.copy()

    # choose the same variables as your screenshot
    cols = []
    for c in ["number_of_reviews", "review_scores_rating", "response_rate"]:
        if c in df.columns:
            cols.append(c)

    if len(cols) < 2:
        print("[WARN] Not enough columns for correlation heatmap.")
        return

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(corr.values)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticklabels(cols)

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Figure 3: Correlation:\nRatings, Reviews, and Responsiveness")
    fig.colorbar(im, ax=ax, label="Correlation Coefficient")

    _save(fig, out_dir, "Fig05_correlation_heatmap.png")


def plot_ratings_vs_reviews_bubble(listings: pd.DataFrame, out_dir: str):
    df = listings.dropna(subset=["review_scores_rating", "number_of_reviews"]).copy()

    x = pd.to_numeric(df["review_scores_rating"], errors="coerce")
    y = pd.to_numeric(df["number_of_reviews"], errors="coerce")

    rr = pd.to_numeric(df.get("response_rate", np.nan), errors="coerce").fillna(0).clip(0, 100)
    sizes = 15 + (rr * 2.5)

    # Colour by review volume (log-scaled) for meaningful variation
    c = np.log1p(y)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    sc = ax.scatter(x, y, s=sizes, c=c, alpha=0.30, edgecolors="none")

    # Cap y for readability
    ycap = np.nanpercentile(y, 99)
    ax.set_ylim(0, ycap * 1.05)

    ax.set_title("Figure 06: Ratings vs Number of Reviews (Size = Host Response Rate, Colour = Review Volume)")
    ax.set_xlabel("Average Review Rating")
    ax.set_ylabel("Total Number of Reviews")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log(1 + Number of Reviews)")

    _save(fig, out_dir, "Fig06_ratings_vs_reviews_bubble.png")



def plot_price_by_amenity_and_host_segment(listings: pd.DataFrame, out_dir: str):
    df = listings.dropna(subset=["price", "amenities"]).copy()
    df["host_segment"] = host_segment_from_counts(df)

    amenities = ["air conditioning", "free parking", "heating", "kitchen", "tv", "washer", "wifi"]

    # boolean flags for amenities (amenities field often looks like "{...}" string)
    for a in amenities:
        col = f"has_{a.replace(' ', '_')}"
        df[col] = df["amenities"].str.lower().str.contains(rf"\b{re.escape(a)}\b", regex=True)

    seg_order = [
        "Enterprise Portfolio (10+ listings)",
        "Large Portfolio (6–10 listings)",
        "Medium Portfolio (2–5 listings)",
        "Small Portfolio (1 listing)",
    ]

    # compute mean price by amenity and segment
    rows = []
    for a in amenities:
        flag = f"has_{a.replace(' ', '_')}"
        for seg in seg_order:
            mean_price = df.loc[(df["host_segment"] == seg) & (df[flag]), "price"].mean()
            rows.append((a.title(), seg, mean_price))

    out = pd.DataFrame(rows, columns=["Amenity", "Host Segment", "Average Price"]).dropna()

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)

    # grouped bars
    amenity_labels = out["Amenity"].drop_duplicates().tolist()
    x = np.arange(len(amenity_labels))
    width = 0.2

    for i, seg in enumerate(seg_order):
        vals = out.loc[out["Host Segment"] == seg].set_index("Amenity").reindex(amenity_labels)["Average Price"].values
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=seg)

    ax.set_title("Average Nightly Price by Amenity and Host Segment")
    ax.set_xlabel("Amenity")
    ax.set_ylabel("Average Price ($)")
    ax.set_xticks(x)
    ax.set_xticklabels(amenity_labels, rotation=30, ha="right")
    ax.legend(title="Host Segment", ncol=2, frameon=False)

    _save(fig, out_dir, "Fig08_price_by_amenity_and_host_segment.png")


def plot_listing_density_points(listings: pd.DataFrame, neighbourhoods_gdf, out_dir: str):
    df = listings.dropna(subset=["latitude", "longitude", "price"]).copy()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # base map
    neighbourhoods_gdf.boundary.plot(ax=ax, linewidth=0.6)

    sc = ax.scatter(
        df["longitude"], df["latitude"],
        c=df["price"],
        s=12,
        alpha=0.6
    )
    ax.set_title("Sydney Airbnb Listing Density (Individual Points)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label="Listing Price ($)")

    _save(fig, out_dir, "Fig09_listing_density_points.png")


def plot_avg_price_choropleth(listings: pd.DataFrame, neighbourhoods_gdf, out_dir: str):
    df = listings.dropna(subset=["neighbourhood_cleansed", "price"]).copy()

    # Average price per neighbourhood
    avg = (
        df.groupby("neighbourhood_cleansed")["price"]
        .mean()
        .reset_index()
        .rename(columns={"neighbourhood_cleansed": "neighbourhood", "price": "avg_price"})
    )

    gdf = neighbourhoods_gdf.copy()

    # Join on the exact same names (we know it matches 38/38)
    merged = gdf.merge(avg, on="neighbourhood", how="left")

    matched = int(merged["avg_price"].notna().sum())
    total = int(len(merged))
    print(f"[INFO] Choropleth join matched {matched}/{total} areas using neighbourhood_cleansed.")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    merged.plot(
        column="avg_price",
        ax=ax,
        legend=True,
        missing_kwds={"color": "lightgrey", "label": "No data"},
    )
    merged.boundary.plot(ax=ax, linewidth=0.4)

    ax.set_title("Figure 10: Sydney Airbnb Price Distribution by Neighbourhood")
    ax.set_axis_off()


    _save(fig, out_dir, "Fig10_avg_price_choropleth.png")


def run(data_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    listings, reviews, neighbourhoods = load_data(data_dir)
    reviews = clean_reviews(reviews)
    listings = basic_clean_listings(listings)
    # Ensure standardized columns exist for these plots
    listings_std = add_standard_columns(listings)

    # Generate key portfolio figures
    make_price_by_room_type(listings, out_dir)
    make_top_expensive_neighbourhoods(listings, out_dir)
    plot_reviews_by_day_of_week(reviews, out_dir)
    plot_weekday_weekend_share(reviews, out_dir)
    plot_correlation_heatmap(listings_std, out_dir)
    plot_ratings_vs_reviews_bubble(listings_std, out_dir)
    plot_price_by_room_type_violin(listings_std, out_dir)
    plot_price_by_amenity_and_host_segment(listings_std, out_dir)
    plot_listing_density_points(listings_std, neighbourhoods, out_dir)
    plot_avg_price_choropleth(listings_std, neighbourhoods, out_dir)
    # You can add more figures here based on your notebook questions.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
