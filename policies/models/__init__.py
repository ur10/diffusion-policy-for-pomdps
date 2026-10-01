from .bc_policy_rnn import BCPolicy_RNN_MLP as BCPolicy_RNN_MLP
from .bc_policy_diffusion import BCPolicy_Diffusion as BCPolicy_Diffusion
from .bc_policy_dit import BCPolicy_DiT as BCPolicy_DiT
from .bc_policy_ldp import BCPolicy_LDP as BCPolicy_LDP
from .bc_policy_ldp import BCPolicy_LDP as BCPolicy_PTP
try:
    from .bc_policy_jepa import BCPolicy_JEPA as BCPolicy_JEPA
except ImportError:
    BCPolicy_JEPA = None

try:
    from .bc_policy_videomae import BCPolicy_VideoMAE as BCPolicy_VideoMAE
except ImportError:
    BCPolicy_VideoMAE = None

AGENT_CLASSES = {
    "Policy_BC_RNN": BCPolicy_RNN_MLP,
    "Policy_BC_Diffusion": BCPolicy_Diffusion,
    "Policy_BC_DiT": BCPolicy_DiT,
    "Policy_BC_LDP": BCPolicy_LDP,
    "Policy_BC_PTP": BCPolicy_PTP,
}

if BCPolicy_JEPA is not None:
    AGENT_CLASSES["Policy_BC_JEPA"] = BCPolicy_JEPA

if BCPolicy_VideoMAE is not None:
    AGENT_CLASSES["Policy_BC_VideoMAE"] = BCPolicy_VideoMAE

from enum import Enum


class AGENT_ARCHS(str, Enum):
    # inherit from str to allow comparison with str
    Memory = BCPolicy_RNN_MLP.ARCH
