from adaptive_strategy import ExecutionGuard, VectorizedRegimeEngine


class RegimeDetector:
    """
    Thin wrapper kept for backward compatibility with the rest of the app.
    """

    def __init__(self, df, benchmark_df=None):
        self.df = df.copy()
        self.benchmark_df = benchmark_df.copy() if benchmark_df is not None and len(benchmark_df) > 0 else None

    def detect_regimes(self):
        regime_frame = VectorizedRegimeEngine(self.df, benchmark_df=self.benchmark_df).build_regime_frame()
        return ExecutionGuard(regime_frame).build_execution_frame()
