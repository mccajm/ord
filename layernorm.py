import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops


ln = tf.load_op_library('layer_norm_fused_op.so')

#This line is needed so TensorFlow can infer the shape of the output.
#This may not be required (may even raise error) if you are using newer version of TensorFlow.
ops.RegisterShape("LayerNormCustom")(common_shapes.call_cpp_shape_fn)

#register gradients for auto-differentiation.
@ops.RegisterGradient("LayerNormCustom")
def _LayerNormCustomGrad(op, grad):
    return [ln.layer_norm_backprop_custom(
        op.inputs[0], grad, op.get_attr("epsilon"))]

