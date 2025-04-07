import shap
import matplotlib.pyplot as plt
from datetime import datetime

def generate_shap_plot_image(model, X):
    # SHAP объяснение
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)
    
    # 💡 expected_value — берем первый (scalar)
    expected_value = explainer.expected_value[0]

    # Сохраняем график
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"shap_plot_{timestamp}.png"

    fig = plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        expected_value,
        shap_values[0],
        feature_names=X.columns,
        max_display=10,
        show=False
    )
    fig.savefig(path, format="png", bbox_inches="tight")
    plt.close(fig)

    return path
