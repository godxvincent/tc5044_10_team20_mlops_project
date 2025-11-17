#!/usr/bin/env python3
"""
Script para validar drift detection en CI/CD.

Este script ejecuta drift detection y falla si se detecta drift significativo
o si hay una degradación importante en el rendimiento del modelo.

Uso:
    python scripts/check_drift_ci.py
    python scripts/check_drift_ci.py --max-performance-drop 15
    python scripts/check_drift_ci.py --fail-on-significant-drift
"""

import argparse
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configurar dynaconf
os.environ.setdefault("ENV_FOR_DYNACONF", "local")

from mlops.pipeline import MLPipeline  # noqa: E402


def evaluate_drift_results(results, max_performance_drop=10.0, fail_on_significant_drift=True):
    """
    Evalúa resultados de drift y determina si deben fallar el check.

    Args:
        results: Diccionario con resultados de drift por escenario
        max_performance_drop: Porcentaje máximo de drop en performance permitido (default: 10%)
        fail_on_significant_drift: Si True, falla si hay drift significativo (p-value < 0.05)

    Returns:
        tuple: (should_fail, failure_reasons)
    """
    should_fail = False
    failure_reasons = []

    for scenario_name, result in results.items():
        if "error" in result:
            should_fail = True
            failure_reasons.append(f"❌ {scenario_name}: Error durante evaluación - {result['error']}")
            continue

        # Verificar drift estadístico
        drift_metrics = result.get("drift_metrics", {})
        all_features_result = drift_metrics.get("_all_features", {})

        if fail_on_significant_drift:
            drift_detected = all_features_result.get("drift_detected", False)
            severity = all_features_result.get("severity", "unknown")
            p_value = all_features_result.get("p_value", float("inf"))

            if drift_detected and severity == "significant":
                should_fail = True
                failure_reasons.append(
                    f"❌ {scenario_name}: Drift significativo detectado "
                    f"(p-value={p_value:.4f}, severity={severity})"
                )

        # Verificar degradación de performance
        comparison = result.get("comparison", {})
        for metric_name, comp_data in comparison.items():
            drop_percent = comp_data.get("drop_percent", 0.0)

            if drop_percent > max_performance_drop:
                should_fail = True
                failure_reasons.append(
                    f"❌ {scenario_name}: Degradación de {metric_name} excede el umbral "
                    f"({drop_percent:.2f}% > {max_performance_drop}%)"
                )

    return should_fail, failure_reasons


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Validar drift detection en CI/CD")
    parser.add_argument(
        "--data-file",
        type=str,
        default="turkish_music_emotion_modified.csv",
        help="Nombre del archivo CSV en data/external/",
    )
    parser.add_argument(
        "--max-performance-drop",
        type=float,
        default=10.0,
        help="Porcentaje máximo de drop en performance permitido (default: 10.0)",
    )
    parser.add_argument(
        "--fail-on-significant-drift",
        action="store_true",
        default=True,
        help="Falla si se detecta drift significativo (p-value < 0.05) (default: True)",
    )
    parser.add_argument(
        "--no-fail-on-drift",
        action="store_true",
        help="No falla por drift significativo, solo por degradación de performance",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Omitir entrenamiento (requiere modelo ya entrenado)",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Omitir evaluación (requiere modelo ya entrenado y evaluado)",
    )

    args = parser.parse_args()

    # Determinar si debemos fallar por drift significativo
    fail_on_drift = args.fail_on_significant_drift and not args.no_fail_on_drift

    print("=" * 80)
    print("🔍 DRIFT DETECTION CI/CD CHECK")
    print("=" * 80)
    print("📊 Configuración:")
    print(f"   - Max performance drop permitido: {args.max_performance_drop}%")
    print(f"   - Fallar por drift significativo: {fail_on_drift}")
    print(f"   - Data file: {args.data_file}")
    print()

    try:
        # Ejecutar pipeline
        pipeline = MLPipeline()
        pipeline.load_data_step(args.data_file)
        pipeline.clean_up_data_step()
        pipeline.feature_engineering_step()

        if not args.skip_train:
            print("🎯 Entrenando modelo...")
            pipeline.train_step()
            print("   ✅ Modelo entrenado")
        else:
            print("⏭️  Omitiendo entrenamiento")

        if not args.skip_evaluate:
            print("📊 Evaluando modelo...")
            pipeline.evaluate_step()
            print("   ✅ Modelo evaluado")
        else:
            print("⏭️  Omitiendo evaluación")

        # Ejecutar drift detection
        print("🔍 Ejecutando drift detection...")
        drift_results = pipeline.drift_monitoring_step()
        print(f"   ✅ {len(drift_results)} escenarios evaluados")
        print()

        # Evaluar resultados
        should_fail, failure_reasons = evaluate_drift_results(
            drift_results,
            max_performance_drop=args.max_performance_drop,
            fail_on_significant_drift=fail_on_drift,
        )

        # Mostrar resumen
        print("=" * 80)
        print("📋 RESUMEN DE RESULTADOS")
        print("=" * 80)

        for scenario_name, result in drift_results.items():
            if "error" in result:
                print(f"\n❌ {scenario_name}: Error - {result['error']}")
                continue

            drift_metrics = result.get("drift_metrics", {})
            all_features = drift_metrics.get("_all_features", {})
            performance = result.get("performance_metrics", {})
            comparison = result.get("comparison", {})

            print(f"\n📋 Escenario: {scenario_name}")
            print(f"   Drift detectado: {all_features.get('drift_detected', False)}")
            print(f"   Severidad: {all_features.get('severity', 'unknown')}")
            print(f"   P-value promedio: {all_features.get('p_value', 'N/A'):.4f}")

            if performance:
                print("   Performance con drift:")
                print(f"      - Accuracy: {performance.get('accuracy', 0):.4f}")
                print(f"      - F1: {performance.get('f1', 0):.4f}")

            if comparison:
                print("   Comparación con baseline:")
                for metric_name, comp_data in comparison.items():
                    drop_pct = comp_data.get("drop_percent", 0.0)
                    status = "✅" if drop_pct <= args.max_performance_drop else "❌"
                    print(
                        f"      {status} {metric_name}: "
                        f"{comp_data.get('baseline', 0):.4f} → {comp_data.get('current', 0):.4f} "
                        f"({drop_pct:+.2f}%)"
                    )

        print()
        print("=" * 80)

        if should_fail:
            print("❌ CHECK FALLIDO - Se detectaron problemas críticos:")
            print()
            for reason in failure_reasons:
                print(f"   {reason}")
            print()
            print("=" * 80)
            sys.exit(1)
        else:
            print("✅ CHECK EXITOSO - No se detectaron problemas críticos")
            print("=" * 80)
            sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
