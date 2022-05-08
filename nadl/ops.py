from ops.cpu import CPUOps
from ops.gpu import GPUOps
from src.utils import sameDeviceCheck


def opsHandler(self, value, op):
    if sameDeviceCheck(self, value):
        op_handler = CPUOps if self.device == "cpu" else GPUOps
        return op_handler.__getattribute__(op)(self, value)
    else:
        raise TypeError("For now, only tensors on the same device can be added")
