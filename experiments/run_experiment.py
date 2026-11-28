
import csv
import time
import statistics
from pathlib import Path

from src.benchmarks import get_benchmark_functions
from src.co_algorithm import cheetah_optimizer


# ==========================================
# RUTAS DEL PROYECTO
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ==========================================
# CONFIGURACION DE LAS FUNCIONES
# ==========================================

funciones = get_benchmark_functions(dim_nd=10)


# ==========================================
# CONFIGURACIONES DEL ALGORITMO CO
# ==========================================

configuraciones = [
    {
        "nombre": "Original",
        "n": 6,
        "m": 2,
        "alpha_scale": 0.0001,
        "r_check_scale": 1.0,
        "use_improvement_rule": False
    },
    {
        "nombre": "Poblacion media",
        "n": 10,
        "m": 2,
        "alpha_scale": 0.0001,
        "r_check_scale": 1.0,
        "use_improvement_rule": False
    },
    {
        "nombre": "Poblacion alta",
        "n": 20,
        "m": 2,
        "alpha_scale": 0.0001,
        "r_check_scale": 1.0,
        "use_improvement_rule": False
    },
    {
        "nombre": "Mayor movimiento",
        "n": 6,
        "m": 2,
        "alpha_scale": 0.0005,
        "r_check_scale": 1.0,
        "use_improvement_rule": False
    },
    {
        "nombre": "Menor escala de ataque",
        "n": 6,
        "m": 2,
        "alpha_scale": 0.0001,
        "r_check_scale": 0.5,
        "use_improvement_rule": False
    }
]


# ==========================================
# PARAMETROS GENERALES
# ==========================================

MAX_FES = 3000
REPETICIONES = 5

resultados = []

total_pruebas = (
    len(funciones) *
    len(configuraciones) *
    REPETICIONES
)

prueba_actual = 0


# ==========================================
# EJECUCION DE LOS EXPERIMENTOS
# ==========================================

for num_funcion, (nombre, datos) in enumerate(funciones.items()):

    print("\n========================================")
    print("FUNCION:", datos["name"])
    print("========================================")

    for num_config, config in enumerate(configuraciones, start=1):

        print("\nConfiguracion:", config["nombre"])

        for repeticion in range(1, REPETICIONES + 1):

            prueba_actual += 1

            # Semilla diferente en cada ejecucion
            semilla = (
                10000
                + num_funcion * 1000
                + num_config * 10
                + repeticion
            )

            inicio = time.perf_counter()

            resultado = cheetah_optimizer(
                fobj=datos["fobj"],
                lb=datos["lb"],
                ub=datos["ub"],
                dim=datos["dim"],
                n=config["n"],
                m=config["m"],
                max_fes=MAX_FES,
                alpha_scale=config["alpha_scale"],
                r_check_scale=config["r_check_scale"],
                use_improvement_rule=config["use_improvement_rule"],
                seed=semilla
            )

            tiempo = time.perf_counter() - inicio

            # Guardar resultados de cada prueba
            fila = {
                "funcion": datos["name"],
                "configuracion": config["nombre"],
                "repeticion": repeticion,
                "n": config["n"],
                "m": config["m"],
                "max_fes": MAX_FES,
                "alpha_scale": config["alpha_scale"],
                "r_check_scale": config["r_check_scale"],
                "seed": semilla,
                "best_cost": resultado["best_cost"],
                "fes": resultado["fes"],
                "iterations": resultado["iterations"],
                "tiempo_segundos": tiempo
            }

            resultados.append(fila)

            print(
                f"Prueba {prueba_actual}/{total_pruebas} | "
                f"Repeticion: {repeticion} | "
                f"Resultado: {resultado['best_cost']:.10f} | "
                f"Tiempo: {tiempo:.4f}s"
            )


# ==========================================
# GUARDAR RESULTADOS INDIVIDUALES
# ==========================================

campos = [
    "funcion",
    "configuracion",
    "repeticion",
    "n",
    "m",
    "max_fes",
    "alpha_scale",
    "r_check_scale",
    "seed",
    "best_cost",
    "fes",
    "iterations",
    "tiempo_segundos"
]

with open(
    RESULTS_DIR / "resultados_experimentos.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    escritor = csv.DictWriter(
        archivo,
        fieldnames=campos
    )

    escritor.writeheader()
    escritor.writerows(resultados)


# ==========================================
# CALCULAR RESUMEN DE RESULTADOS
# ==========================================

resumen = []

for nombre in funciones:

    datos_funcion = [
        r for r in resultados
        if r["funcion"] == funciones[nombre]["name"]
    ]

    for config in configuraciones:

        grupo = [
            r for r in datos_funcion
            if r["configuracion"] == config["nombre"]
        ]

        valores = [r["best_cost"] for r in grupo]

        resumen.append({
            "funcion": funciones[nombre]["name"],
            "configuracion": config["nombre"],
            "mejor_resultado": min(valores),
            "peor_resultado": max(valores),
            "promedio": statistics.mean(valores),
            "desviacion_estandar": statistics.pstdev(valores),
            "tiempo_promedio": statistics.mean(
                r["tiempo_segundos"] for r in grupo
            )
        })


# ==========================================
# GUARDAR RESUMEN
# ==========================================

campos_resumen = [
    "funcion",
    "configuracion",
    "mejor_resultado",
    "peor_resultado",
    "promedio",
    "desviacion_estandar",
    "tiempo_promedio"
]

with open(
    RESULTS_DIR / "resumen_experimentos.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as archivo:

    escritor = csv.DictWriter(
        archivo,
        fieldnames=campos_resumen
    )

    escritor.writeheader()
    escritor.writerows(resumen)


# ==========================================
# FINALIZACION
# ==========================================

print("\n========================================")
print("EXPERIMENTOS FINALIZADOS")
print("========================================")

print("Funciones evaluadas:", len(funciones))
print("Configuraciones:", len(configuraciones))
print("Repeticiones:", REPETICIONES)
print("Total de pruebas:", len(resultados))

print("\nArchivos generados:")
print(RESULTS_DIR / "resultados_experimentos.csv")
print(RESULTS_DIR / "resumen_experimentos.csv")
