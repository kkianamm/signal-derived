from .medtsllm import MedTsLLM
from .medtsllm_image_fusion import MedTsLLMImageFusion
from .medtsllm_qformer import MedTsLLMQFormer
from .medtsllm_convnext_diagnostic_router import MedTsLLMConvNeXtDiagnosticRouter
from .gpt4ts import GPT4TS

from .dlinear import DLinear
from .FEDformer import FEDformer
from .PatchTST import PatchTST
from .TimesNet import TimesNet

model_lookup = {
    "timellm": MedTsLLM,
    "medtsllm": MedTsLLM,
    "medtsllm_image_fusion": MedTsLLMImageFusion,
    "medtsllm_qformer": MedTsLLMQFormer,
    "medtsllm_convnext_diagnostic_router": MedTsLLMConvNeXtDiagnosticRouter,
    "gpt4ts": GPT4TS,
    "dlinear": DLinear,
    "fedformer": FEDformer,
    "patchtst": PatchTST,
    "timesnet": TimesNet,
}
