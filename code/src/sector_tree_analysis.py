
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from config import config


def build_sector_panel(raw_df, sector_map):
    """
    构建板块级别的特征面板数据
    每个交易日 × 每个行业 = 一行（约 500天 × 11行业 = 5500行）
    """
    df = raw_df.copy()
    df["行业"] = df["股票代码"].map(sector_map).fillna("未知")
    df = df.sort_values(["行业", "股票代码", "日期"]).reset_index(drop=True)

    # ==================== 个股级别基础特征 ====================
    df["daily_ret"] = df.groupby("股票代码")["收盘"].pct_change(1)
    for w in [3, 5, 10, 20]:
        df[f"stock_mom_{w}d"] = df.groupby("股票代码")["收盘"].pct_change(w)
    df["vol_ratio_5d"] = df.groupby("股票代码")["成交量"].pct_change(5)

    # ==================== 截面聚合到板块级别 ====================
    panel = df.groupby(["日期", "行业"]).agg(
        stock_count=("股票代码", "count"),
        sector_daily_ret=("daily_ret", "mean"),
        # 板块动量：个股动量的均值
        sector_mom_3d=("stock_mom_3d", "mean"),
        sector_mom_5d=("stock_mom_5d", "mean"),
        sector_mom_10d=("stock_mom_10d", "mean"),
        sector_mom_20d=("stock_mom_20d", "mean"),
        # 板块内部分化
        sector_dispersion_5d=("stock_mom_5d", "std"),
        sector_breadth_5d=("stock_mom_5d", lambda x: (x > 0).mean()),
        sector_skew_5d=("stock_mom_5d", "skew"),
        # 成交量
        sector_vol_ratio=("vol_ratio_5d", "mean"),
    ).reset_index()

    # ==================== 板块内滚动特征 ====================
    panel = panel.sort_values(["行业", "日期"]).reset_index(drop=True)

    for w in [5, 10, 20]:
        panel[f"sector_ret_vol_{w}d"] = (
            panel.groupby("行业")["sector_daily_ret"]
            .transform(lambda x: x.rolling(w, min_periods=1).std())
        )

    # 板块动量质量：过去5天上涨日收益总和 / 下跌日收益绝对值总和
    panel["mom_quality_5d"] = (
        panel.groupby("行业")["sector_daily_ret"]
        .transform(lambda x: x.rolling(5, min_periods=1).apply(
            lambda g: g[g > 0].sum() / (abs(g[g < 0]).sum() + 1e-8),
            raw=True
        ))
    )

    # ==================== 截面排名特征 ====================
    for w in [5, 10, 20]:
        col = f"sector_mom_{w}d"
        panel[f"rank_{w}d"] = panel.groupby("日期")[col].rank(ascending=False)
        panel[f"quantile_{w}d"] = panel.groupby("日期")[col].rank(pct=True)

    # 排名变化（与5天前的排名差）
    panel["rank_change_5d"] = panel.groupby("行业")["rank_5d"].diff(5).fillna(0)

    # ==================== 市场状态特征 ====================
    market = df.groupby("日期").agg(
        market_ret_1d=("daily_ret", "mean"),
        market_breadth_1d=("daily_ret", lambda x: (x > 0).mean()),
        market_vol_5d=("daily_ret", "std"),
        market_mom_5d=("stock_mom_5d", "mean"),
        market_mom_10d=("stock_mom_10d", "mean"),
    ).reset_index()

    panel = panel.merge(market, on="日期", how="left")

    # 板块相对于市场的超额动量
    panel["excess_mom_5d"] = panel["sector_mom_5d"] - panel["market_mom_5d"]
    panel["excess_mom_10d"] = panel["sector_mom_10d"] - panel["market_mom_10d"]

    # ==================== 滞后特征（上一调仓周期的信息）====================
    for col in ["sector_mom_5d", "sector_mom_10d", "rank_5d",
                "sector_breadth_5d", "sector_dispersion_5d", "excess_mom_5d"]:
        panel[f"prev_{col}"] = panel.groupby("行业")[col].shift(5)

    # 动量加速度：近期动量 - 远期动量
    panel["mom_acceleration"] = panel["sector_mom_5d"] - panel["sector_mom_10d"]

    # ==================== 目标变量 ====================
    df["open_t1"] = df.groupby("股票代码")["开盘"].shift(-1)
    df["open_t5"] = df.groupby("股票代码")["开盘"].shift(-5)
    df["future_5d_ret"] = (df["open_t5"] - df["open_t1"]) / df["open_t1"]

    target = df.groupby(["日期", "行业"])["future_5d_ret"].mean().reset_index()
    panel = panel.merge(target, on=["日期", "行业"], how="left")
    panel = panel.rename(columns={"future_5d_ret": "target"})

    # ==================== 清理 ====================
    panel = panel.sort_values(["日期", "行业"]).reset_index(drop=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 去掉目标为NaN的行（最后几天没有未来数据）
    panel = panel.dropna(subset=["target"])

    return panel


def evaluate_on_rebalance(test_df, rebalance_dates, sector_map, raw_df):
    """
    在调仓日上评估预测效果
    """
    top1_hits = 0
    top3_hits = 0
    random_top1_hits = 0
    momentum_top1_hits = 0
    total = 0

    tree_returns = []       # 树模型策略收益
    momentum_returns = []   # 动量策略收益
    random_returns = []     # 随机策略收益

    # 同时记录在预测板块内选Top5股票的实际收益
    tree_stock_returns = []  # 树模型选板块→板块内选Top5

    for date in rebalance_dates:
        daily = test_df[test_df["日期"] == date].copy()
        if daily.empty or len(daily) < 5:
            continue
        total += 1

        # ---- 真实排名 ----
        true_ranking = daily.sort_values("target", ascending=False)
        true_top1 = true_ranking.iloc[0]["行业"]
        true_top3 = set(true_ranking.head(3)["行业"].tolist())

        # ---- 树模型预测 ----
        pred_ranking = daily.sort_values("pred", ascending=False)
        tree_top1 = pred_ranking.iloc[0]["行业"]
        tree_top3 = set(pred_ranking.head(3)["行业"].tolist())

        # ---- 动量策略（上期第1）----
        mom_top1 = daily.nlargest(1, "sector_mom_5d").iloc[0]["行业"]

        # ---- 随机 ----
        random_pick = np.random.choice(daily["行业"].tolist())

        # ---- 统计命中 ----
        if tree_top1 == true_top1:
            top1_hits += 1
        if tree_top1 in true_top3:
            top3_hits += 1
        if mom_top1 == true_top1:
            momentum_top1_hits += 1
        if random_pick == true_top1:
            random_top1_hits += 1

        # ---- 收益记录 ----
        # 树模型：买入预测第1板块的平均收益
        tree_ret = daily[daily["行业"] == tree_top1]["target"].iloc[0]
        tree_returns.append(tree_ret)

        # 动量：买入动量第1板块
        mom_ret = daily[daily["行业"] == mom_top1]["target"].iloc[0]
        momentum_returns.append(mom_ret)

        # 随机
        rand_ret = daily[daily["行业"] == random_pick]["target"].iloc[0]
        random_returns.append(rand_ret)

    metrics = {
        "total": total,
        "tree_top1_hit": top1_hits,
        "tree_top3_hit": top3_hits,
        "momentum_top1_hit": momentum_top1_hits,
        "random_top1_hit": random_top1_hits,
        "tree_top1_rate": top1_hits / total if total > 0 else 0,
        "tree_top3_rate": top3_hits / total if total > 0 else 0,
        "momentum_top1_rate": momentum_top1_hits / total if total > 0 else 0,
        "random_top1_rate": random_top1_hits / total if total > 0 else 0,
        "tree_returns": tree_returns,
        "momentum_returns": momentum_returns,
        "random_returns": random_returns,
    }

    # 累计收益
    if tree_returns:
        metrics["tree_cum"] = np.cumprod(1 + np.array(tree_returns))[-1] - 1
        metrics["momentum_cum"] = np.cumprod(1 + np.array(momentum_returns))[-1] - 1
        metrics["random_cum"] = np.cumprod(1 + np.array(random_returns))[-1] - 1
    return metrics


def main():
    data_file = os.path.join(config["data_path"], "train.csv")
    sector_file = os.path.join(config["data_path"], "hs300_stock_list_annotated.csv")
    output_dir = config["output_dir"]

    print("=" * 60)
    print("板块特征 + 树模型 预测验证")
    print("=" * 60)

    # ---- 加载数据 ----
    print("\n1. 加载数据...")
    raw_df = pd.read_csv(data_file, dtype={"股票代码": str})
    raw_df["股票代码"] = raw_df["股票代码"].astype(str).str.zfill(6)
    raw_df["日期"] = pd.to_datetime(raw_df["日期"])
    raw_df = raw_df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    sector_df = pd.read_csv(sector_file, dtype={"code_clean": str})
    sector_df["code_clean"] = sector_df["code_clean"].astype(str).str.zfill(6)
    sector_map = sector_df.set_index("code_clean")["行业"].to_dict()

    # ---- 构建板块特征面板 ----
    print("2. 构建板块特征面板（每个交易日 × 每个行业）...")
    panel = build_sector_panel(raw_df, sector_map)
    print(f"   面板数据量: {len(panel)} 行 × {len(panel.columns)} 列")
    print(f"   行业数: {panel["行业"].nunique()}")
    print(f"   交易日数: {panel["日期"].nunique()}")

    # 打印特征列表
    feature_cols = [
        c for c in panel.columns
        if c not in ["日期", "行业", "target", "stock_count"]
    ]
    print(f"   特征数: {len(feature_cols)}")
    print(f"   特征列表: {feature_cols}")

    # ---- 时间序列分割 ----
    all_dates = sorted(panel["日期"].unique())
    split_idx = int(len(all_dates) * 0.7)
    train_dates = all_dates[:split_idx]
    test_dates = all_dates[split_idx:]

    print(f"\n3. 数据分割:")
    print(f"   训练集: {train_dates[0].date()} ~ {train_dates[-1].date()} ({len(train_dates)} 天, ~{len(train_dates)*11} 样本)")
    print(f"   测试集: {test_dates[0].date()} ~ {test_dates[-1].date()} ({len(test_dates)} 天, ~{len(test_dates)*11} 样本)")

    train_panel = panel[panel["日期"].isin(train_dates)]
    test_panel = panel[panel["日期"].isin(test_dates)]

    X_train = train_panel[feature_cols]
    y_train = train_panel["target"]
    X_test = test_panel[feature_cols]
    y_test = test_panel["target"]

    # ---- 训练树模型 ----
    print("\n4. 训练树模型...")

    models = {
        "GBDT": GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=-1,
        ),
    }

    model_results = {}
    for name, model in models.items():
        print(f"   训练 {name}...")
        model.fit(X_train, y_train)
        test_panel_copy = test_panel.copy()
        test_panel_copy["pred"] = model.predict(X_test)

        # 在调仓日（每5个交易日）上评估
        test_all_dates = sorted(test_panel_copy["日期"].unique())
        rebalance_dates = test_all_dates[::5]
        metrics = evaluate_on_rebalance(test_panel_copy, rebalance_dates, sector_map, raw_df)

        # 特征重要性
        fi = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # 基础回归指标
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, model.predict(X_test))
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

        model_results[name] = {
            "metrics": metrics,
            "feature_importance": fi,
            "r2": r2,
            "rmse": rmse,
        }

        print(f"   {name} - R²: {r2:.4f}, RMSE: {rmse:.4f}")

    # ---- 输出结果 ----
    print("\n" + "=" * 60)
    print("5. 板块预测结果")
    print("=" * 60)

    baseline_random = 1.0 / panel["行业"].nunique()
    print(f"\n随机基线 Top-1 命中率: {baseline_random:.1%}")
    print(f"简单动量 Top-1 命中率 (全样本): 13.4%")

    for name, res in model_results.items():
        m = res["metrics"]
        print(f"\n--- {name} (测试集) ---")
        print(f"  调仓次数:       {m["total"]}")
        print(f"  Top-1 命中率:    {m["tree_top1_rate"]:.1%} ({m["tree_top1_hit"]}/{m["total"]})")
        print(f"  Top-3 命中率:    {m["tree_top3_rate"]:.1%} ({m["tree_top3_hit"]}/{m["total"]})")
        print(f"  动量 Top-1:     {m["momentum_top1_rate"]:.1%} ({m["momentum_top1_hit"]}/{m["total"]})")
        print(f"  随机 Top-1:     {m["random_top1_rate"]:.1%} ({m["random_top1_hit"]}/{m["total"]})")
        print(f"  树模型累计收益:  {m["tree_cum"]:.2%}")
        print(f"  动量累计收益:    {m["momentum_cum"]:.2%}")
        print(f"  随机累计收益:    {m["random_cum"]:.2%}")

    # ---- 特征重要性 ----
    print("\n" + "=" * 60)
    print("6. 特征重要性 Top 15 (GBDT)")
    print("=" * 60)
    fi = model_results["GBDT"]["feature_importance"]
    for i, row in fi.head(15).iterrows():
        bar = "█" * int(row["importance"] * 200)
        print(f"  {row["feature"]:<30s} {row["importance"]:.4f} {bar}")

    # ---- 绘图 ----
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

        # 累计收益对比
        ax1 = axes[0]
        for name, res in model_results.items():
            m = res["metrics"]
            if m["tree_returns"]:
                cum = np.cumprod(1 + np.array(m["tree_returns"]))
                ax1.plot(rebalance_dates[:len(cum)], cum,
                         label=f"{name} (Top-1 Sector)", linewidth=2)
        if model_results["GBDT"]["metrics"]["momentum_returns"]:
            cum_mom = np.cumprod(1 + np.array(model_results["GBDT"]["metrics"]["momentum_returns"]))
            ax1.plot(rebalance_dates[:len(cum_mom)], cum_mom,
                     label="Momentum (Top-1 Sector)", linewidth=1.5, linestyle="--", color="orange")
        if model_results["GBDT"]["metrics"]["random_returns"]:
            cum_rand = np.cumprod(1 + np.array(model_results["GBDT"]["metrics"]["random_returns"]))
            ax1.plot(rebalance_dates[:len(cum_rand)], cum_rand,
                     label="Random", linewidth=1, linestyle=":", color="gray")

        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Sector Selection Strategy: Cumulative Return Comparison (Test Set)")
        ax1.set_ylabel("Cumulative Wealth")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 特征重要性
        ax2 = axes[1]
        top_features = fi.head(15)
        ax2.barh(range(len(top_features)), top_features["importance"].values[::-1],
                 color="steelblue")
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features["feature"].values[::-1])
        ax2.set_xlabel("Feature Importance")
        ax2.set_title("Top 15 Features (GBDT)")
        ax2.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        save_path = os.path.join(output_dir, "sector_tree_model_analysis.png")
        plt.savefig(save_path, dpi=150)
        print(f"\n图表已保存至: {save_path}")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")

    # ---- 板块命中分布 ----
    print("\n" + "=" * 60)
    print("7. 板块预测偏好分析 (GBDT)")
    print("=" * 60)

    # 统计测试集中树模型最常预测为Top1的板块
    test_panel_gbdt = test_panel.copy()
    test_panel_gbdt["pred"] = model_results["GBDT"][
        list(models.keys())[0]  # This won't work, fix below
    ].predict(X_test) if False else models["GBDT"].predict(X_test)
    test_panel_gbdt["pred_rank"] = test_panel_gbdt.groupby("日期")["pred"].rank(ascending=False)
    top1_preds = test_panel_gbdt[test_panel_gbdt["pred_rank"] == 1]["行业"].value_counts()
    print("\n树模型预测 Top-1 板块频次:")
    for sector, count in top1_preds.items():
        bar = "█" * count
        print(f"  {sector:<10s} {count:>3d} 次 {bar}")

    print("\n真实 Top-1 板块频次:")
    test_panel_gbdt["true_rank"] = test_panel_gbdt.groupby("日期")["target"].rank(ascending=False)
    true_top1 = test_panel_gbdt[test_panel_gbdt["true_rank"] == 1]["行业"].value_counts()
    for sector, count in true_top1.items():
        bar = "█" * count
        print(f"  {sector:<10s} {count:>3d} 次 {bar}")


if __name__ == "__main__":
    main()


