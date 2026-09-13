from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class RecurrenceVisualization:
    """Deterministic demonstration-field renderer with configurable parameters."""
    def __init__(self, seed=404, alpha=0.1745, N=240, steps=1600, out_dir=".", batch_output_size=40):
        self.seed, self.alpha, self.N, self.steps = seed, alpha, N, steps
        self.out_dir, self.batch_output_size = Path(out_dir), batch_output_size
        self.out_dir.mkdir(parents=True, exist_ok=True)
    @staticmethod
    def laplacian(A):
        return (np.roll(A,1,0)+np.roll(A,-1,0)+np.roll(A,1,1)+np.roll(A,-1,1)-4*A)
    def generate_field(self):
        rng=np.random.default_rng(self.seed); x=np.linspace(-1,1,self.N); y=np.linspace(-1,1,self.N); X,Y=np.meshgrid(x,y)
        H=0.002*rng.normal(size=(self.N,self.N)); H += np.exp(-((X-.13)**2+(Y+.07)**2)/.0012)
        theta=np.arctan2(Y,X); r=np.sqrt(X**2+Y**2)
        scaffold=.010*np.sin(7*theta+5.5*r)+.008*np.cos(6*theta-4.7*r)+.004*np.sin(2*np.pi*(X+Y))
        for t in range(self.steps):
            local=self.laplacian(H)
            perturbation=np.tanh(local)
            H=np.tanh(H + 0.11*local + scaffold - self.alpha*perturbation)
        return H
    def render(self, filename=None):
        H=self.generate_field(); filename=filename or "recurrence_field_demo.png"; out=self.out_dir/filename
        fig,ax=plt.subplots(figsize=(7,7)); ax.imshow(H,cmap="plasma",origin="lower",extent=[-1,1,-1,1]); ax.set_title("Recurrence Field Demonstration"); ax.set_xticks([]); ax.set_yticks([]); fig.savefig(out,dpi=220,bbox_inches="tight"); plt.close(fig); return out
    def render_batch(self,seeds):
        old=self.seed; outputs=[]
        for seed in seeds: self.seed=seed; outputs.append(self.render(f"recurrence_field_seed_{seed}_alpha_{self.alpha}_N_{self.N}_steps_{self.steps}.png"))
        self.seed=old; return outputs
