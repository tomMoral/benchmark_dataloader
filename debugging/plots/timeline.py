from benchopt.plotting.base import BasePlot


class Plot(BasePlot):
    """Per-batch time over iteration index (scatter plot).

    Shows *when* stalls start (e.g. batch 50) and which solvers avoid them.
    """

    name = "Batch time timeline"
    type = "scatter"
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
                "x": df_solver["objective_batch_idx"].tolist(),
                "y": df_solver["objective_batch_time_ms"].tolist(),
                "label": solver,
                **style,
            })

        return plots

    def get_metadata(self, df, dataset, objective):
        return {
            "title": f"Batch time timeline\nData: {dataset}",
            "xlabel": "Batch index",
            "ylabel": "Batch time (ms)",
        }
