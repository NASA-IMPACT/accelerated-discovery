from .inference import Inferencer
from typing import List

class AlignScore:
    def __init__(
            self, 
            model: str, 
            batch_size: int, 
            device: int, 
            ckpt_path: str, 
            evaluation_mode='nli_sp',
            granularity='sentence',
            verbose=True
    ) -> None:
        self.model = Inferencer(
            ckpt_path=ckpt_path, 
            model=model,
            batch_size=batch_size, 
            device=device,
            granularity=granularity,
            verbose=verbose
        )
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str], op1: str = "max", op2: str = "max") -> List[float]:
        return self.model.nlg_eval(contexts, claims, op1, op2)[1].tolist()