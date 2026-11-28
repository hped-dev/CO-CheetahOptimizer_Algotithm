from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df_runs = pd.read_csv(RESULTS_DIR / "resultados_experimentos.csv")
    df_summary = pd.read_csv(RESULTS_DIR / "resumen_experimentos.csv")

    print("=== MEJOR CONFIGURACIÓN POR FUNCIÓN (SEGÚN PROMEDIO) ===")
    mejores = df_summary.sort_values("promedio").groupby("funcion", as_index=False).first()
    print(mejores[["funcion", "configuracion", "promedio", "desviacion_estandar", "mejor_resultado"]])

    # 1. Gráfica comparativa del promedio por configuración en cada función
    for func, grupo in df_runs.groupby("funcion"):
        plt.figure(figsize=(9, 5))
        orden_cfg = grupo.groupby("configuracion")["best_cost"].mean().sort_values().index
        datos_box = [grupo[grupo["configuracion"] == cfg]["best_cost"].values for cfg in orden_cfg]

        plt.boxplot(datos_box, labels=orden_cfg, patch_artist=True)
        plt.yscale("log")
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Mejor costo (escala log)")
        plt.title(f"Sensibilidad de parámetros en {func}")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = FIGURES_DIR / f"boxplot_{func.lower().replace(' ', '_')}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()

    # 2. Gráfica resumen de tiempo promedio por configuración
    pivot_tiempo = df_summary.pivot(index="funcion", columns="configuracion", values="tiempo_promedio")
    ax = pivot_tiempo.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Tiempo promedio (s)")
    ax.set_title("Tiempo de ejecución promedio por función y configuración")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tiempo_promedio.png", dpi=300)
    plt.close()

    print(f"\nGráficas guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()