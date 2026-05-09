import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

PLOTS = {
    "idle_ram_mb": "RAM nach Modell-Laden (MB)",
    "idle_vram_mb": "VRAM nach Modell-Laden (MB)",
    "vram_peak_mb": "VRAM Maximum (MB)",
    "inf_mean_s": "Inferenzzeit Durchschnitt (s)",
    "ram_max_mb": "RAM Maximum (MB)",
    "ram_avg_mb": "RAM Durchschnitt (MB)",
    "cpu_%": "CPU Auslastung (%)",
    "eer_%": "EER (%)",
    "eer_schwelle": "EER-Schwelle",
    "min_dcf": "Minimum DCF",
    "tar@far=1%": "TAR bei FAR = 1 %",
    "tar@far=5%": "TAR bei FAR = 5 %",
}

def all_plots_generate(results_path: str) -> None:
    json_path = Path(results_path) / "summary.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f) # Liste von Dict-Objekten, jedes Dict enthält die Ergebnisse für ein Modell, z.B. {"modell": "Pyannote", "idle_ram_mb": 500, "eer_%": 5.0, ...}
    modules = [d["modell"] for d in data] # Liste der Modellnamen
    plot_dir = Path(results_path) / "plots"
    plot_dir.mkdir(exist_ok=True)

    sns.set_theme(style="darkgrid", palette="Set2") # ändern je nach Geschmack, z.B. "darkgrid", "white", "dark", "ticks" oder eigene Farben mit sns.set_palette()

    for measurement, title in PLOTS.items():
        values = [d.get(measurement) for d in data]
        plt.figure(figsize=(8, len(data) * 0.5))
        sns.barplot(x=values, y=modules)
        plt.title(title, fontsize=13, fontweight="bold") # pad = Abstand zum Plot, vielleicht ergänzen
        plt.tight_layout()
        filename = measurement
        plt.savefig(plot_dir / f"{filename}.png", dpi=200, bbox_inches="tight") # dpi für höhere Auflösung, bbox_inches="tight" für weniger Rand
        plt.close()

if __name__ == "__main__":
    all_plots_generate(str(Path(__file__).parent.parent / "results"))
