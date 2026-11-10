"""Redraw Marked5 numerical figures from immutable, saved CSV evidence.

This module performs no training, inference, uncertainty resampling, network
access, or source-data reconstruction. Redraws are numerical equivalents, not
the exact manuscript images retained in publication/reference. Run from any cwd:

python scripts/publication_export.py --repo-root . --output publication/redraws
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [7, 42, 123, 999, 2024]
CLIENTS = ["Client1_NW_Arid", "Client2_Indo_Gangetic", "Client3_NE_Himalaya",
           "Client4_South_Peninsular", "Client5_Central_India"]
CLIENT_LABELS = ["C1 NW", "C2 Indo-Gangetic", "C3 NE/Himalaya", "C4 South", "C5 Central"]
GLOBAL = ["CAAA_GLOBAL_FULL", "FEDAVG_GLOBAL_FULL", "FEDPROX_GLOBAL_FULL",
          "CENTRAL_GLOBAL_FULL", "raw_pooled_ET_FULL_latest", "local_ET_FULL_latest",
          "client_normalized_pooled_ET_FULL_latest"]
GLOBAL_LABELS = ["CAAA global", "FedAvg global", "FedProx global", "Central neural",
                 "ET pooled", "ET local", "ET client-normalized pooled"]
PERSONAL = ["CAAA_PERSONAL_FULL", "DITTO_PERSONAL_FULL", "LOCAL_PERSONAL_FULL",
            "GRAPH_ORIGINAL_PERSONAL_FULL", "GRAPH_ENRICHED_PERSONAL_FULL", "GRAPH_UNIFORM_PERSONAL_FULL"]
PERSONAL_LABELS = ["CAAA personal", "Ditto personal", "Local neural", "Graph original",
                   "Graph enriched", "Graph uniform"]
POLICIES = ["LOCAL_PRIVATE", "LOCAL_STATS", "FEDAVG_PRIVATE_K1", "FEDAVG_STATS_K1",
            "FEDAVG_STATS_K5", "FEDAVG_STATS_K20", "CAAA_STATS_K1", "CENTRAL_POOL_STATS"]
POLICY_LABELS = ["Local / own", "Local / shared", "FedAvg / own K1", "FedAvg / shared K1",
                 "FedAvg / shared K5", "FedAvg / shared K20", "CA-AdapFL / shared K1", "Central pooled / shared"]
COLORS = ["#21618c", "#b9770e", "#148f77", "#884ea0", "#c0392b", "#566573", "#2874a6"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class Evidence:
    def __init__(self, repo: Path, accepted: Path, sharing: Path):
        self.repo, self.accepted, self.sharing = repo, accepted, sharing
        self.used = {}

    def display_path(self, path: Path) -> str:
        for base, name in [(self.accepted, "benchmarks/accepted"), (self.sharing, "benchmarks/sharing")]:
            if path.is_relative_to(base):
                return name + "/" + path.relative_to(base).as_posix()
        return path.relative_to(self.repo).as_posix()

    def csv(self, relative: str, sharing: bool = False) -> pd.DataFrame:
        path = (self.sharing if sharing else self.accepted) / relative
        require(path.is_file(), f"Missing saved evidence: {path}")
        frame = pd.read_csv(path, float_precision="round_trip")
        self.used[self.display_path(path)] = {"sha256": sha(path), "rows": len(frame)}
        return frame

    def source(self, name: str) -> pd.DataFrame:
        return self.csv("figures/revision_v2/source_data/" + name)


class Export:
    def __init__(self, output: Path, evidence: Evidence):
        self.output, self.evidence = output, evidence
        self.figures = []

    def save(self, fig, figure_id: str, caption: str, source_names: list[str]) -> None:
        fig.suptitle(figure_id + " — numerical redraw", fontsize=14)
        fig.text(.015, .012, "Numerically equivalent redraw; exact embedded manuscript original is retained separately.", fontsize=9)
        fig.tight_layout(rect=(0, .045, 1, .96))
        files = {}
        for extension in ("png", "svg"):
            path = self.output / (figure_id + "." + extension)
            fig.savefig(path, dpi=200, facecolor="white", metadata={"Creator": "CAAdapFL saved-evidence exporter"} if extension == "svg" else None)
            files[path.name] = sha(path)
        plt.close(fig)
        caption_path = self.output / (figure_id + "_caption.txt")
        caption_path.write_text(caption + "\n\nNumerical redraw, not an exact reconstruction of the embedded manuscript image.\n", encoding="utf-8")
        files[caption_path.name] = sha(caption_path)
        resolved_sources = [("benchmarks/accepted/figures/revision_v2/source_data/" + name.removeprefix("accepted/"))
                            if name.startswith("accepted/") else
                            ("benchmarks/sharing/evaluation/" + name.removeprefix("sharing/"))
                            for name in source_names]
        self.figures.append({"id": figure_id, "render_role": "numerically_equivalent_redraw",
                             "caption": caption, "sources": resolved_sources, "files_sha256": files})


def validate_seed_panel(frame: pd.DataFrame, models: list[str]) -> None:
    require(not frame.duplicated(["model", "seed"]).any(), "Duplicate model/seed in seed panel")
    for model in models:
        require(sorted(frame.loc[frame.model == model, "seed"].astype(int)) == SEEDS,
                "Expected the five paired optimizer seeds for " + model)


def seed_panels(export: Export, frame: pd.DataFrame, models: list[str], labels: list[str], figure_id: str, source: str) -> None:
    validate_seed_panel(frame, models)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric, xlabel in zip(axes, ["R2", "RMSE_mm"], ["Macro R² (= NSE)", "Macro RMSE (mm)"]):
        for i, model in enumerate(models):
            values = frame.loc[frame.model == model].sort_values("seed")[metric].to_numpy()
            ax.scatter(values, i + np.linspace(-.09, .09, 5), facecolors="none", edgecolors=COLORS[i % len(COLORS)], s=26)
            ax.errorbar(values.mean(), i, xerr=values.std(ddof=1), fmt="D", color=COLORS[i % len(COLORS)], capsize=4, ms=6)
        if metric == "R2":
            ax.axvline(0, color="#555", linestyle=":")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.2)
        ax.margins(x=.12)
    axes[0].set_yticks(range(len(models)), labels)
    axes[0].invert_yaxis()
    export.save(fig, figure_id, "Fixed FULL " + ("global and classical" if figure_id == "Figure05" else "personal neural") +
                " identities. Open points are the five paired optimizer seeds; diamonds and whiskers show mean ± sample seed SD, not confidence intervals. "
                "Metrics average clients equally. Global and personal checkpoints were selected separately on VAL.", [source])


def observed_forecast(export: Export, raw: pd.DataFrame, per: pd.DataFrame) -> None:
    raw = raw[raw.model.isin(["CAAA_GLOBAL_FULL", "training_seasonal_mean"])].copy()
    require(not raw.duplicated(["model", "seed", "client", "target_month"]).any(), "Duplicate forecast keys")
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, client, label in zip(axes.flat, CLIENTS, CLIENT_LABELS):
        neural = raw[(raw.model == "CAAA_GLOBAL_FULL") & (raw.client == client)]
        clim = raw[(raw.model == "training_seasonal_mean") & (raw.client == client)]
        require(len(neural) == 480 and len(clim) == 480, f"Expected 480 repeated forecasts per plotted identity/client: {client}")
        require(neural.target_month.nunique() == 96, "Expected 96 target months")
        clim = clim[clim.seed == SEEDS[0]]
        require(len(clim) == 96, "Expected one climatology forecast per target month")
        ax.scatter(neural.y, neural.prediction, s=10, color=COLORS[0], alpha=.35, label="CAAA global: five seeds")
        ax.scatter(clim.y, clim.prediction, s=22, facecolors="none", edgecolors=COLORS[1], marker="s", label="TRAIN climatology")
        q = per[(per.model == "CAAA_GLOBAL_FULL") & (per.client == client)].heavy80_training_threshold_mm.unique()
        require(len(q) == 1, "TRAIN q80 changes between paired seeds")
        ax.axvline(q[0], color="#777", linestyle=":", label="TRAIN q80")
        low = min(0., float(neural.prediction.min()), float(clim.prediction.min()))
        high = max(float(neural.y.max()), float(neural.prediction.max()), float(clim.prediction.max()))
        pad = .04 * (high-low)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], "--", color="#555", linewidth=1)
        ax.set_xlim(low-pad, high+pad); ax.set_ylim(low-pad, high+pad)
        ax.set_title(label); ax.set_xlabel("Observed rainfall (mm)"); ax.set_ylabel("Forecast rainfall (mm)")
        ax.grid(alpha=.15)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    export.save(fig, "Figure07", "Observed versus forecast regional monthly rainfall, 2017–2024. Each panel shows all 480 CAAA global FULL seed-specific forecasts (96 dates × five seeds), "
                "and the 96 distinct TRAIN monthly climatology forecasts. Dashed line: 1:1; dotted vertical line: local gradient-TRAIN q80. Panels have separate scales. "
                "Negative forecasts are retained. Repeated seeds add forecasts, not independent observations.", ["accepted/raw_predictions_ALL.csv", "accepted/per_client_ALL.csv"])


def graph_panel(export: Export, graph: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, variant in zip(axes, ["original", "enriched", "uniform"]):
        part = graph[graph.variant == variant]
        require(len(part) == 25 and not part.duplicated(["client", "reference_client"]).any(), "Expected a 5×5 graph matrix")
        matrix = part.pivot(index="client", columns="reference_client", values="input_affinity").loc[CLIENTS, CLIENTS].to_numpy()
        require(np.isfinite(matrix).all() and (matrix >= 0).all() and (matrix <= 1).all(), "Affinity outside [0,1]")
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(variant.title()); ax.set_xticks(range(5), ["C1", "C2", "C3", "C4", "C5"])
        ax.set_yticks(range(5), ["C1", "C2", "C3", "C4", "C5"])
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", color="white" if matrix[i,j] > .7 else "#17202a", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=.7, label="TRAIN affinity")
    export.save(fig, "Figure08", "Fixed original, enriched and uniform TRAIN input affinities on a shared 0–1 scale. TRAIN targets run 1983-01 through 2007-11 (299 per client). "
                "These structural summaries use privileged aligned TRAIN anomalies/descriptors; they are neither causal effects nor forecast skill. Row-normalized trainer weights remain in the source CSV.", ["accepted/05_training_graphs.csv"])


def development_panel(export: Export, grid: pd.DataFrame, rounds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    lr = grid[grid.stage == "LR"].sort_values("lr")
    require(len(lr) == 2, "Expected two learning-rate candidates")
    axes[0,0].plot(lr.lr, lr.validation_macro_R2, "o-", color=COLORS[0])
    axes[0,0].set_xscale("log"); axes[0,0].set_xticks(lr.lr, [str(x) for x in lr.lr]); axes[0,0].set_xlabel("Learning rate")
    axes[0,0].set_ylabel("Validation macro R²"); axes[0,0].set_title("(a) FULL FedAvg global, seed 42")
    mus = grid[grid.stage.str.startswith("MU_")]
    for i, (stage, group) in enumerate(mus.groupby("stage", sort=False)):
        group = group.sort_values("mu")
        axes[0,1].plot(group.mu, group.validation_macro_R2, "o-", color=COLORS[i], label=stage.removeprefix("MU_").replace("_", " "))
        chosen = group[group.selected]
        axes[0,1].scatter(chosen.mu, chosen.validation_macro_R2, facecolors="none", edgecolors="#17202a", s=130)
    axes[0,1].set_xscale("log"); axes[0,1].set_xticks([.001,.01,.1], ["0.001","0.01","0.1"])
    axes[0,1].set_xlabel("Proximal μ"); axes[0,1].set_ylabel("Validation macro R²"); axes[0,1].set_title("(b) Six proximal methods"); axes[0,1].legend(fontsize=8)
    c = grid[grid.stage == "COEFFICIENTS"].copy()
    require(len(c) == 16 and np.allclose(c[["alpha","beta","gamma"]].sum(axis=1), 1), "Expected the sixteen simplex candidates")
    c["x"] = c.beta + .5*c.gamma; c["y"] = np.sqrt(3)*.5*c.gamma
    axes[1,0].plot([0,1,.5,0], [0,0,np.sqrt(3)/2,0], color="#777")
    anchor = np.isclose(c.alpha,.3) & np.isclose(c.beta,.3) & np.isclose(c.gamma,.4)
    normal = c[~anchor]; anchor_rows = c[anchor]
    im=axes[1,0].scatter(normal.x, normal.y, c=normal.validation_macro_R2, s=90, cmap="viridis", vmin=c.validation_macro_R2.min(), vmax=c.validation_macro_R2.max())
    axes[1,0].scatter(anchor_rows.x, anchor_rows.y, c=anchor_rows.validation_macro_R2, marker="D", s=100, cmap="viridis", vmin=c.validation_macro_R2.min(), vmax=c.validation_macro_R2.max())
    chosen = c[c.selected]; axes[1,0].scatter(chosen.x, chosen.y, facecolors="none", edgecolors="black", s=170)
    for x,y,label in [(0,0,"α=1"),(1,0,"β=1"),(.5,np.sqrt(3)/2,"γ=1")]:axes[1,0].annotate(label,(x,y),xytext=(0,12 if y else -16),textcoords="offset points",ha="center")
    axes[1,0].set_title("(c) CAAA aggregation coefficients; diamond = anchor"); axes[1,0].set_axis_off(); axes[1,0].margins(.15)
    fig.colorbar(im,ax=axes[1,0],shrink=.75,label="Validation macro R²")
    for i, stage in enumerate(["LR","MU_caaa","COEFFICIENTS"]):
        selected = grid[(grid.stage == stage) & grid.selected]
        require(len(selected) == 1, "Expected one selected candidate in "+stage)
        part = rounds[(rounds.stage == stage) & (rounds.job_sha256 == selected.iloc[0].job_sha256) & (rounds.identity == selected.iloc[0].identity)].sort_values("round")
        require(len(part) == 60, "Expected sixty saved validation rounds for "+stage)
        axes[1,1].plot(part["round"],part.validation_macro_R2,color=COLORS[i],label=stage)
        checkpoint=part[part.selected_checkpoint]
        axes[1,1].scatter(checkpoint["round"],checkpoint.validation_macro_R2,facecolors="none",edgecolors=COLORS[i],s=110)
    axes[1,1].set_xlabel("Saved round"); axes[1,1].set_ylabel("Validation macro R²"); axes[1,1].set_title("(d) Selected candidates and checkpoints"); axes[1,1].legend(fontsize=9)
    export.save(fig, "Figure09", "Seed-42 development grids on January 2011–November 2016 VAL. (a) Shared learning rate. (b) Six method-specific proximal μ grids. "
                "(c) Fifteen quarter-step simplex triples plus the (0.3,0.3,0.4) anchor; color uses its own validation scale. Rings mark selected candidates. "
                "(d) All sixty saved validation rounds for the selected LR, CAAA μ and coefficient candidates. Validation was reused for tuning/checkpoint selection; these are developmental scores, not independent accuracy estimates.", ["accepted/06_validation_candidates_ALL.csv", "accepted/06_validation_rounds_ALL.csv"])


def anchor_panel(export: Export, ci: pd.DataFrame) -> None:
    names = ["Climate weight", "Performance weight", "Proximal term", "Auxiliary loss", "Climate embedding"]
    fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    for ax,metric,label in zip(axes,["macro_R2","macro_RMSE_mm"],["Δ macro R² (anchor − removal)","Δ macro RMSE (mm; anchor − removal)"]):
        rows=ci[ci.metric==metric].sort_values("comparison_order")
        require(len(rows)==5 and (rows.block_length_months==12).all(), "Expected five fixed-anchor block12 intervals")
        for i,row in enumerate(rows.itertuples()):
            ax.plot([row.CI95_low,row.CI95_high],[i,i],color=COLORS[0],linewidth=2)
            ax.scatter(row.delta_a_minus_b,i,color=COLORS[0],marker="D",s=45)
        ax.axvline(0,color="#555",linestyle="--");ax.set_xlabel(label);ax.grid(axis="x",alpha=.2)
    axes[0].set_yticks(range(5),names);axes[0].invert_yaxis()
    export.save(fig,"Figure10","Fixed (0.3,0.3,0.4) anchor minus each separate removal. Points and lines are saved paired differences and marginal unadjusted 95% circular block12 intervals. "
                "Positive R²/negative RMSE favors the anchor. The selected (1,0,0) primary is a distinct identity. Intervals are not simultaneous; block6/24 sensitivities remain in companion evidence.",["accepted/04_anchor_ablations.csv"])


def precision_recall(export: Export, pr: pd.DataFrame, per: pd.DataFrame) -> None:
    fig,axes=plt.subplots(2,3,figsize=(13,9))
    for ax,client,label in zip(axes.flat,CLIENTS,CLIENT_LABELS):
        for i,model in enumerate(GLOBAL[:3]):
            rows=pr[(pr.model==model)&(pr.client==client)&(pr.seed==42)].sort_values("point_order")
            require(len(rows)>1,"Missing saved PR coordinates")
            require(((rows[["precision","recall"]]>=0)&(rows[["precision","recall"]]<=1)).all().all(),"PR coordinates outside [0,1]")
            ax.plot(rows.recall,rows.precision,color=COLORS[i],label=GLOBAL_LABELS[i],linewidth=1.5)
        support=per[(per.model==GLOBAL[0])&(per.client==client)&(per.seed==42)].iloc[0]
        ax.axhline(support.actual_heavy_n/support.n,color="#777",linestyle="-.",label="Observed prevalence")
        ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title(label+f" ({int(support.actual_heavy_n)}/96)")
        ax.set_xlabel("Recall");ax.set_ylabel("Precision");ax.grid(alpha=.2)
    axes.flat[-1].axis("off");handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc="center",frameon=False)
    export.save(fig,"Figure11","Separate client precision–recall curves for the three FULL global identities at fixed illustrative seed 42. Scores are predicted heavy-event probabilities. "
                "Events exceed the local gradient-TRAIN q80; dash-dot line is observed evaluation prevalence. No pooled-client curve is shown. This realization is not an event-scale flood-warning validation.",["accepted/08_client_PR_seed42.csv","accepted/per_client_ALL.csv"])


def heavy_panel(export: Export, per: pd.DataFrame) -> None:
    fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True,sharey="row")
    for column,(models,labels) in enumerate([(GLOBAL[:3],GLOBAL_LABELS[:3]),(PERSONAL,PERSONAL_LABELS)]):
        offsets=np.linspace(-.27,.27,len(models))
        for row,(metric,axislabel,percent) in enumerate([("heavy_recall","Recall (%)",True),("heavy_average_precision","Average precision (%)",True),("tail80_RMSE_mm","Tail RMSE (mm)",False)]):
            ax=axes[row,column]
            for m,(model,label) in enumerate(zip(models,labels)):
                for c,client in enumerate(CLIENTS):
                    part=per[(per.model==model)&(per.client==client)].sort_values("seed")
                    require(sorted(part.seed.astype(int))==SEEDS,"Missing heavy-event seed/client panel: "+model+" "+client)
                    vals=part[metric].to_numpy()*(100 if percent else 1)
                    require(np.isfinite(vals).all(),"Undefined plotted heavy diagnostic")
                    if percent:require((vals>=0).all() and (vals<=100).all(),"Probability diagnostic outside [0,100]")
                    x=c+offsets[m];mean=float(vals.mean());sd=float(vals.std(ddof=1));low=mean-sd;high=mean+sd
                    ax.scatter(x+np.linspace(-.022,.022,5),vals,s=8,alpha=.45,color=COLORS[m])
                    ax.plot([x,x],[max(0,low) if percent else low,min(100,high) if percent else high],color=COLORS[m],linewidth=1.5)
                    ax.scatter(x,mean,s=30,marker="D",color=COLORS[m],label=label if c==0 and row==0 else None)
                    if percent and high>100:ax.annotate("",xy=(x,100),xytext=(x,96),arrowprops={"arrowstyle":"-|>","color":COLORS[m],"lw":1})
                    if percent and low<0:ax.annotate("",xy=(x,0),xytext=(x,4),arrowprops={"arrowstyle":"-|>","color":COLORS[m],"lw":1})
            ax.set_ylabel(axislabel);ax.set_xticks(range(5),["C1","C2","C3","C4","C5"]);ax.grid(axis="y",alpha=.2)
            if percent:ax.set_ylim(-2,104)
            if row==0:ax.set_title("Global FULL" if column==0 else "Personal FULL");ax.legend(fontsize=8,loc="lower left",ncol=2)
    export.save(fig,"Figure12","Per-client heavy-event recall, average precision and rainfall tail RMSE for fixed FULL global/personal identities. Small points: five seeds; diamonds and whiskers: mean ± sample seed SD, not confidence intervals. "
                "Recall uses probability ≥0.5; AP uses continuous probabilities; tail RMSE scores observed months exceeding local TRAIN q80. Proportion whiskers stop at 0/100% with arrows when SD extends beyond those bounds. "
                "Seed values themselves remain in [0,100%]; repeated seeds do not add observed events.",["accepted/per_client_ALL.csv"])


def communication_panel(export: Export, principal: pd.DataFrame, seeds: pd.DataFrame, cost: pd.DataFrame) -> None:
    summary=principal.set_index("model");means=cost.groupby("model").mean(numeric_only=True)
    require((cost.groupby("model").logical_required_payload_bytes.nunique()==1).all(),"Policy required bytes vary between seeds")
    fig,ax=plt.subplots(figsize=(13.5,7));markers=["o","s","^","D","v","P","X","h"]
    for i,model in enumerate(POLICIES):
        part=seeds[(seeds.model==model)&(seeds.endpoint=="fixed60")].sort_values("seed")
        require(sorted(part.seed.astype(int))==SEEDS,"Missing primary sharing seeds")
        x=float(means.loc[model,"logical_required_payload_bytes"])/2**20;y=float(summary.loc[model,"RMSE_mm_mean"])
        require(np.isclose(part.RMSE_mm.mean(),y,rtol=1e-12,atol=1e-10),"Primary sharing means mismatch")
        ax.plot([x,x],[part.RMSE_mm.min(),part.RMSE_mm.max()],color="#a9bacb",linewidth=2)
        ax.scatter(x,y,marker=markers[i],s=80,label=POLICY_LABELS[i],color=COLORS[i%len(COLORS)])
    ax.axhline(float(summary.loc["TRAIN_CLIMATOLOGY","RMSE_mm_mean"]),color="#333",linestyle="--",label="TRAIN climatology")
    ax.set_xscale("symlog",linthresh=.01);ax.set_xlabel("Required numeric payload (MiB; symmetric-log scale retains zero)")
    ax.set_ylabel("Macro RMSE (mm)");ax.grid(alpha=.2);ax.margins(x=.12,y=.15);ax.legend(loc="center left",bbox_to_anchor=(1.01,.5),fontsize=9)
    export.save(fig,"Figure13","Information-sharing study, fixed tick60. Points show mean macro RMSE; ranges are optimizer-seed minimum–maximum, not confidence intervals. Own/shared denotes local/shared TRAIN input scaling. "
                "K1/K5/K20 is synchronization cadence in local ticks. Required numeric payload sums recipients and includes federated initial/final model delivery; central pooled payload includes engineered examples/final delivery. "
                "Transport and research audit traffic are excluded. Dashed line is TRAIN climatology. No network-latency or compute saving is implied.",["sharing/principal_comparison.csv","sharing/macro_per_seed_ALL_endpoints.csv","sharing/fit_cost_40.csv"])


def checkpoint_panel(export: Export, summary: pd.DataFrame) -> None:
    fig,ax=plt.subplots(figsize=(12,7))
    for i,model in enumerate(POLICIES):
        values=[]
        for endpoint in ["fixed60","VAL20_40_60"]:
            part=summary[(summary.model==model)&(summary.endpoint==endpoint)]
            require(len(part)==1,"Missing common checkpoint endpoint: "+model)
            values.append(float(part.RMSE_mm_mean.iloc[0]))
        ax.plot(values,[i,i],color="#a9bacb",linewidth=2)
        ax.scatter(values[0],i,marker="D",s=55,color=COLORS[0],label="Fixed tick60" if i==0 else None)
        ax.scatter(values[1],i,marker="o",s=65,facecolors="white",edgecolors=COLORS[1],label="VAL choice among 20/40/60" if i==0 else None)
    ax.set_yticks(range(8),POLICY_LABELS);ax.invert_yaxis();ax.set_xlabel("Macro RMSE (mm)");ax.legend();ax.grid(axis="x",alpha=.2)
    export.save(fig,"FigureIS1","Common VAL checkpoint sensitivity for the same fitted policies. Primary fixed tick60 and secondary common selection among ticks20/40/60 are shown for global/local paths. "
                "Personal diagnostic paths remain separately identified in model_summary_ALL_endpoints.csv. Checkpoint communication is separately priced; optimizer repeats and endpoint forecasts do not add independent observations.",["sharing/model_summary_ALL_endpoints.csv"])


def main(argv=None) -> None:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="Fresh, nonexistent directory; resolved from current cwd")
    parser.add_argument("--accepted-root",type=Path,help="Optional evidence-root override for inspection/smoke testing")
    parser.add_argument("--sharing-root",type=Path,help="Optional evidence-root override for inspection/smoke testing")
    args=parser.parse_args(argv)
    repo=args.repo_root.resolve();accepted=(args.accepted_root or repo/"benchmarks/accepted").resolve();sharing=(args.sharing_root or repo/"benchmarks/sharing").resolve();output=args.output.resolve()
    require(not output.exists(),"Refusing to overwrite an existing output directory: "+str(output))
    for protected in [accepted,sharing,repo/"publication/reference"]:
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),"Output overlaps protected evidence/reference: "+str(protected))
    evidence=Evidence(repo,accepted,sharing)
    # Preflight all CSVs before creating any output. No optional IMD inputs are read.
    frames={name:evidence.source(name) for name in ["01_global_and_classical.csv","02_personal.csv","raw_predictions_ALL.csv","per_client_ALL.csv","05_training_graphs.csv","06_validation_candidates_ALL.csv","06_validation_rounds_ALL.csv","04_anchor_ablations.csv","08_client_PR_seed42.csv"]}
    share={name:evidence.csv("evaluation/"+name,sharing=True) for name in ["principal_comparison.csv","macro_per_seed_ALL_endpoints.csv","fit_cost_40.csv","model_summary_ALL_endpoints.csv"]}
    # Exact client and identity keys are read from saved, frozen exports.
    global CLIENTS
    clients=sorted(frames["per_client_ALL.csv"].client.unique())
    require(len(clients)==5,"Expected five fixed clients");CLIENTS=clients
    available=frames["02_personal.csv"].model.unique().tolist()
    require(set(available)==set(PERSONAL),"Expected the six fixed personal identities")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none","figure.facecolor":"white"})
    output.mkdir(parents=True,exist_ok=False);export=Export(output,evidence)
    seed_panels(export,frames["01_global_and_classical.csv"],GLOBAL,GLOBAL_LABELS,"Figure05","accepted/01_global_and_classical.csv")
    seed_panels(export,frames["02_personal.csv"],PERSONAL,PERSONAL_LABELS,"Figure06","accepted/02_personal.csv")
    observed_forecast(export,frames["raw_predictions_ALL.csv"],frames["per_client_ALL.csv"])
    graph_panel(export,frames["05_training_graphs.csv"])
    development_panel(export,frames["06_validation_candidates_ALL.csv"],frames["06_validation_rounds_ALL.csv"])
    anchor_panel(export,frames["04_anchor_ablations.csv"])
    precision_recall(export,frames["08_client_PR_seed42.csv"],frames["per_client_ALL.csv"])
    heavy_panel(export,frames["per_client_ALL.csv"])
    communication_panel(export,share["principal_comparison.csv"],share["macro_per_seed_ALL_endpoints.csv"],share["fit_cost_40.csv"])
    checkpoint_panel(export,share["model_summary_ALL_endpoints.csv"])
    manifest={"status":"RENDERED_NUMERICAL_REDRAWS_REQUIRING_VISUAL_REVIEW","original_images_role":"Exact originals retained in publication/reference; these PNG/SVG exports are distinct numerical equivalents.",
              "no_training_inference_network_resampling_or_IMD_input_access":True,"source_csvs":evidence.used,"figures":export.figures,
              "runtime":{"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__},"exporter_sha256":sha(Path(__file__).resolve())}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (output/"README.txt").write_text("These are numerical redraws from saved evidence, not exact embedded manuscript originals.\n"+
        "No models, checkpoints, statistics or source products were recomputed. Seed means/SDs are descriptive.\n"+
        "Inspect each PNG/SVG visually; manifest status records rendering only. IMD input values are never read or copied by this exporter.\n",encoding="utf-8")
    print(f"Rendered {len(export.figures)} numerical figures (PNG + SVG) from {len(evidence.used)} saved CSVs into {output}")


if __name__=="__main__":
    main()
