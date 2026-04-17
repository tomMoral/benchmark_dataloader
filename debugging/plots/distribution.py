from benchopt.plotting.base import BasePlot


class Plot(BasePlot):
    """Batch time distribution per solver (box plot).

    Distinguishes a uniformly slow solver from one with occasional stalls
    (high p95 vs high median).
    """

    name = "Batch time distribution"
    type = "boxplot"
    options = {
        "dataset": ...,
        "objective": ...,
    }

    def plot(self, df, dataset, objective):
        df = df.query(
            "dataset_name == @dataset and objective_name == @objective"
        )
        plots = []
        for solver, df_solver in df.groupby("solver_name"):
            style = self.get_style(solver)

            plots.append({
                "x": [solver],
                "y": [df_solver["objective_batch_time_ms"].tolist()],
                "label": solver,
                "color": style["color"],
            })

        return plots

    def get_metadata(self, df, dataset, objective):
        return {
            "title": f"Batch time distribution\nData: {dataset}",
            "xlabel": "Solver",
            "ylabel": "Batch time (ms)",
        }
