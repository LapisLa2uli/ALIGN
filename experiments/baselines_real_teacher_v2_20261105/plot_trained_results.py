"""Plot completed trained trials without modifying selection or scores."""
from common import RUN, read


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strict = read(RUN / "trained_results.json")["results"]
    units = read(RUN / "note_unit_results.json")["results"]
    sizes = [0, 10, 20, 40]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for model, color, name in [
        ("polytune", "#2367a6", "PolyTune"),
        ("laddersym", "#bd5b18", "LadderSym"),
    ]:
        values = [100 * strict[model][str(n)]["strict"]["f1"] for n in sizes]
        note_values = [100 * units[model][str(n)]["f1"] for n in sizes]
        for ax, points in zip(axes, [values, note_values]):
            ax.plot(sizes, points, marker="o", linewidth=2, color=color, label=name)
            ax.axhline(points[0], color=color, alpha=.25, linestyle="--")
        for n in sizes[1:]:
            if not strict[model][str(n)]["passed_validation_stability_guard"]:
                for ax, source in zip(axes, [values, note_values]):
                    ax.scatter(n, source[sizes.index(n)], marker="x", s=85, color="#111111", zorder=4)
    for ax, title in zip(axes, ["Exact reference-note range + error type", "Individual reference-note + error type"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training recordings (0 = unchanged weights)")
        ax.set_ylabel("F1 (%)")
        ax.set_xticks(sizes)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle("DataCreate 100–120: actual trained checkpoints", fontsize=14)
    fig.text(.5, -.10,
             "17 scored recordings; AudioEval v9 + MuScriptor pseudo-gold. × = failed validation stability guard.\n"
             "Checkpoints selected using validation only. One seed; previously examined benchmark.",
             ha="center", fontsize=9)
    fig.savefig(RUN / "trained_learning_curve.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("WROTE", RUN / "trained_learning_curve.png", flush=True)


if __name__ == "__main__":
    main()
