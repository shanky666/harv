import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_pie_chart(grades: dict) -> bytes:
    labels = [k for k, v in grades.items() if v > 0]
    sizes = [v for v in grades.values() if v > 0]
    colors = {"good": "#4CAF50", "better": "#8BC34A", "medium": "#FFC107", "reject": "#F44336"}
    clrs = [colors.get(l, "#999") for l in labels]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(sizes, labels=[l.title() for l in labels], colors=clrs,
           autopct="%1.1f%%", startangle=140)
    ax.set_title("Fruit Quality Distribution")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
