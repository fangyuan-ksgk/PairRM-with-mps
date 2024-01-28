from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
@dataclass_json
@dataclass
class BlenderConfig:
    device:str = field(default="mps",
        metadata={"help": "Device, mps, cuda or cpu"}
    )
    use_tqdm:bool = field(default=True,
        metadata={"help": "Use tqdm progress bar"}
    )
    