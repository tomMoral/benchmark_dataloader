import statistics

from benchopt.plotting.base import BasePlot


class Plot(BasePlot):
    """Median throughput per solver (bar chart).

    Summary comparison across solvers with cold/warm ratio annotation.
    A ratio << 1 flags network filesystem bottlenecks.
    """

    name = "Throughput comparison"
    type = "bar_chart"
    options = {
        "dataset": ...,
        "objective": ...,
    }

    def plot(self, df, dataset, objective):
        df = df.query(
            "dataset_name == @dataset and objective_name == @objective"
        )
        plots = []
        throughputs = {}

        for solver, df_solver in df.groupby("solver_name"):
            style = self.get_style(solver)

            # Throughput is repeated on every per-batch row; take one per rep.
            values = (
                df_solver.groupby("idx_rep")[
                    "objective_throughput_samples_per_sec"
                ]
                .first()
                .tolist()
            )
            throughputs[solver] = values

            # Compute cold/warm ratio annotation
            text = ""
            solver_lower = solver.lower().replace(" ", "").replace("-", "")
            for other, other_vals in throughputs.items():
                other_lower = other.lower().replace(" ", "").replace("-", "")
                if (
                    "cold" in solver_lower
                    and "warm" in other_lower
                    and other_lower.replace("warm", "") == solver_lower.replace(
                        "cold", ""
                    )
                ):
                    cold_med = statistics.median(values)
                    warm_med = statistics.median(other_vals)
                    if warm_med > 0:
                        text = f"cold/warm={cold_med / warm_med:.2f}"

            plots.append({
                "y": values,
                "label": solver,
                "color": style["color"],
                "text": text,
            })

        return plots

    def get_metadata(self, df, dataset, objective):
        return {
            "title": f"Throughput by solver\nData: {dataset}",
            "ylabel": "Samples / second",
        }
