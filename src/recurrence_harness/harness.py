import random
from pathlib import Path
from .field import RecurrenceVisualization


class RecurrenceHarness:
    def __init__(
        self,
        alpha: float = 0.01,
        N: int = 256,
        steps: int = 128,
        out_dir: str = "outputs",
        batch_output_size: int = 40,
    ):
        self.alpha = alpha
        self.N = N
        self.steps = steps
        self.out_dir = Path(out_dir)
        self.batch_output_size = batch_output_size

    def single_render(self, seed: int) -> Path:
        viz = RecurrenceVisualization(
            seed=seed,
            alpha=self.alpha,
            N=self.N,
            steps=self.steps,
            out_dir=str(self.out_dir),
            batch_output_size=self.batch_output_size,
        )

        return viz.render()

    def random_seed_sweep(self) -> list[Path]:
        seeds = self.generate_random_seeds(self.batch_output_size)

        viz = RecurrenceVisualization(
            seed=seeds[0],
            alpha=self.alpha,
            N=self.N,
            steps=self.steps,
            out_dir=str(self.out_dir),
            batch_output_size=self.batch_output_size,
        )

        return viz.render_batch(seeds)

    def alpha_sweep(self, seed: int, alpha_values: list[float]) -> list[Path]:
        outputs = []

        for alpha in alpha_values:
            viz = RecurrenceVisualization(
                seed=seed,
                alpha=alpha,
                N=self.N,
                steps=self.steps,
                out_dir=str(self.out_dir),
                batch_output_size=self.batch_output_size,
            )

            filename = f"recurrence_seed_{seed}_alpha_{alpha}_N_{self.N}_steps_{self.steps}.png"
            outputs.append(viz.render(filename=filename))

        return outputs

    def generate_random_seeds(self, count: int) -> list[int]:
        return [random.randint(0, 2**63 - 1) for _ in range(count)]