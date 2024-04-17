# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/combine_joints_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2
from mediapipe.framework.formats import body_rig_pb2 as mediapipe_dot_framework_dot_formats_dot_body__rig__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n:mediapipe/calculators/util/combine_joints_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a*mediapipe/framework/formats/body_rig.proto\"\xaa\x02\n\x1e\x43ombineJointsCalculatorOptions\x12\x12\n\nnum_joints\x18\x01 \x01(\x05\x12O\n\x0ejoints_mapping\x18\x02 \x03(\x0b\x32\x37.mediapipe.CombineJointsCalculatorOptions.JointsMapping\x12\'\n\rdefault_joint\x18\x03 \x01(\x0b\x32\x10.mediapipe.Joint\x1a \n\rJointsMapping\x12\x0f\n\x03idx\x18\x01 \x03(\x05\x42\x02\x10\x01\x32X\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xf9\x91\xe7\xc1\x01 \x01(\x0b\x32).mediapipe.CombineJointsCalculatorOptions')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.combine_joints_calculator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_COMBINEJOINTSCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  _COMBINEJOINTSCALCULATOROPTIONS_JOINTSMAPPING.fields_by_name['idx']._options = None
  _COMBINEJOINTSCALCULATOROPTIONS_JOINTSMAPPING.fields_by_name['idx']._serialized_options = b'\020\001'
  _globals['_COMBINEJOINTSCALCULATOROPTIONS']._serialized_start=156
  _globals['_COMBINEJOINTSCALCULATOROPTIONS']._serialized_end=454
  _globals['_COMBINEJOINTSCALCULATOROPTIONS_JOINTSMAPPING']._serialized_start=332
  _globals['_COMBINEJOINTSCALCULATOROPTIONS_JOINTSMAPPING']._serialized_end=364
# @@protoc_insertion_point(module_scope)
