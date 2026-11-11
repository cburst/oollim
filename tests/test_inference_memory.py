import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'speech'))
from pronunciation.inference_memory import install,ATTENTION_MODULES

def attention(module,name='MultiHeadedAttention'):
 class Attention(torch.nn.Module):
  def forward(self,x):
   self.attn=torch.ones(1,8,4,4);return x
 Attention.__module__,Attention.__name__=module,name
 return Attention()

@pytest.mark.parametrize('module',ATTENTION_MODULES)
def test_inference_releases_attention_maps(module):
 handle=install()
 try:
  m=attention(module).eval();m(torch.zeros(1));assert m.attn is None
  m.train();m(torch.zeros(1));assert isinstance(m.attn,torch.Tensor)
 finally:handle.remove()

def test_other_modules_and_attn_submodules_untouched():
 class Layer(torch.nn.Module):
  def __init__(self):super().__init__();self.attn=torch.nn.Identity()
  def forward(self,x):return self.attn(x)
 Layer.__module__='pxeus.model.powsm.e_branchformer'
 handle=install()
 try:
  layer=Layer().eval();layer(torch.zeros(1));assert isinstance(layer.attn,torch.nn.Identity)
  m=attention('third_party.attention').eval();m(torch.zeros(1));assert isinstance(m.attn,torch.Tensor)
 finally:handle.remove()

def test_vendored_module_path_is_covered_by_its_literal_name():
 """The parametrized test above draws its cases from the tuple under test, so
 it would still pass if the vendored path were dropped from that tuple. This
 one writes the path out, so removing it fails here."""
 assert 'pxeus.espnet_import.attention' in ATTENTION_MODULES
 handle=install()
 try:
  m=attention('pxeus.espnet_import.attention').eval();m(torch.zeros(1));assert m.attn is None
 finally:handle.remove()

def test_tensor_check_spares_an_attn_submodule_on_a_matching_path():
 """A module that passes both the path and the class filter, but whose attn is
 a submodule rather than a tensor, must be left alone. Nothing in the vendored
 attention module is shaped this way today, which is why the tensor check is
 defence in depth, and this test is what keeps it from being dead code."""
 class MultiHeadedAttention(torch.nn.Module):
  def __init__(self):super().__init__();self.attn=torch.nn.Identity()
  def forward(self,x):return self.attn(x)
 MultiHeadedAttention.__module__='pxeus.espnet_import.attention'
 handle=install()
 try:
  m=MultiHeadedAttention().eval();m(torch.zeros(1))
  assert isinstance(m.attn,torch.nn.Identity)
 finally:handle.remove()

def test_installed_phoneticxeus_attention_is_covered(monkeypatch):
 root=ROOT/'speech/resources/phoneticxeus'
 if not (root/'pxeus/espnet_import/attention.py').is_file():pytest.skip('PhoneticXeus source not downloaded')
 # monkeypatch restores sys.path when the test ends, so the asset root does
 # not stay on the path for whatever runs next.
 monkeypatch.syspath_prepend(str(root))
 from pxeus.espnet_import import attention as source
 handle=install()
 try:
  m=source.MultiHeadedAttention(2,8,0.0).eval();x=torch.randn(1,5,8)
  with torch.inference_mode():m(x,x,x,torch.ones(1,1,5,dtype=torch.bool))
  assert type(m).__module__ in ATTENTION_MODULES and m.attn is None
 finally:handle.remove()
